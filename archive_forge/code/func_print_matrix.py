import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def print_matrix(matrix):
    """Print out a matrix for debugging purposes."""
    matrixT = [[] for x in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrixT[j].append(len(str(matrix[i][j])))
    ndigits = [max(x) for x in matrixT]
    for i in range(len(matrix)):
        print(' '.join(('%*s ' % (ndigits[j], matrix[i][j]) for j in range(len(matrix[i])))))