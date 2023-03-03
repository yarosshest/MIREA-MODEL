from openpyxl import load_workbook
import pandas as pd
import numpy as np
import copy
from pdb import set_trace


def pars(c_app):
    wb = load_workbook('Шестаков_ИКБО-06-21_Видеоредакторы.xlsx')
    sheet = wb.active
    weight = [cell.value for cell in sheet['B1':'I1'][0]]
    apps = {}
    for row in sheet['A3':'H' + str(3 + c_app - 1)]:
        apps[row[0].value] = [cell.value for cell in row[1:]]
    return weight, apps


def parsPD():
    appsM = pd.read_excel('Шестаков_ИКБО-06-21_Видеоредакторы.xlsx', index_col=0, header=None, skiprows=2)
    weight = \
    pd.read_excel('Шестаков_ИКБО-06-21_Видеоредакторы.xlsx', index_col=0, header=None, nrows=1).to_numpy().tolist()[0]
    apps = appsM.to_dict('index')
    apps = list(apps.keys())
    matrix = appsM.to_numpy().tolist()
    return weight, apps, matrix


def normalization(apps):
    mx = list(apps.values())
    x_max = [max([x[i] for x in mx]) for i in range(len(mx[0]))]
    vec = mx.copy()
    for i in range(len(vec)):
        for j in range(len(vec[i])):
            vec[i][j] /= x_max[j]
    print(vec)


def saw(w, apps):
    return [round(sum([apps[i][j] * w[j] for j in range(len(apps[i]))]), 3) for i in range(len(apps))]


def mout(w, apps):
    x_max = [max([x[i] for x in apps]) for i in range(len(apps[0]))]
    x_min = [min([x[i] for x in apps]) for i in range(len(apps[0]))]
    matrix = apps.copy()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = round((matrix[i][j] - x_min[j]) / (x_max[j] - x_min[j]), 3)
    return [round(sum([matrix[i][j] * w[j] for j in range(len(matrix[i]))]), 3) for i in range(len(matrix))]


def ranking(self, data):
    return [i + 1 for i in data.argsort()]


def topical(weight, matrix):
    # normalized scores
    row_size = len(matrix)
    column_size = len(matrix[0])
    normalized_decision = np.copy(matrix)
    sqrd_sum = np.zeros(column_size)
    for i in range(row_size):
        for j in range(column_size):
            sqrd_sum[j] += matrix[i][j] ** 2
    for i in range(row_size):
        for j in range(column_size):
            normalized_decision[i, j] = matrix[i][j] / (sqrd_sum[j] ** 0.5)

    # Calculate the weighted normalised decision matrix
    weighted_normalized = np.copy(normalized_decision)
    for i in range(row_size):
        for j in range(column_size):
            weighted_normalized[i, j] *= weight[j]

    # Step 4 Determine the worst alternative and the best alternative

    worst_alternatives = np.zeros(column_size)
    best_alternatives = np.zeros(column_size)
    for i in range(column_size):
        worst_alternatives[i] = min(
            weighted_normalized[:, i])
        best_alternatives[i] = max(weighted_normalized[:, i])

    # /

    worst_distance = np.zeros(row_size)
    best_distance = np.zeros(row_size)

    worst_distance_mat = np.copy(weighted_normalized)
    best_distance_mat = np.copy(weighted_normalized)

    for i in range(row_size):
        for j in range(column_size):
            worst_distance_mat[i][j] = (weighted_normalized[i][j] - worst_alternatives[j]) ** 2
            best_distance_mat[i][j] = (weighted_normalized[i][j] - best_alternatives[j]) ** 2

            worst_distance[i] += worst_distance_mat[i][j]
            best_distance[i] += best_distance_mat[i][j]

    for i in range(row_size):
        worst_distance[i] = worst_distance[i] ** 0.5
        best_distance[i] = best_distance[i] ** 0.5

    # /
    np.seterr(all='ignore')
    worst_similarity = np.zeros(row_size)
    best_similarity = np.zeros(row_size)

    for i in range(row_size):
        # calculate the similarity to the worst condition
        worst_similarity[i] = worst_distance[i] / \
                              (worst_distance[i] + best_distance[i])

        # calculate the similarity to the best condition
        best_similarity[i] = best_distance[i] / \
                             (worst_distance[i] + best_distance[i])

    return best_similarity


if __name__ == '__main__':
    c_app = 4
    methods = ['saw', 'mout', 'topical']

    weight, apps, matrix = parsPD()

    r_saw = saw(weight, copy.deepcopy(matrix))
    r_maut = mout(weight, copy.deepcopy(matrix))
    r_topsis = topical(weight, copy.deepcopy(matrix))

    print('\\', *methods)
    for i in range(len(r_saw)):
        print(apps[i], r_saw[i], r_maut[i], round(r_topsis[i],3))
