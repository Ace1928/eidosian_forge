import numpy as np
from ase import Atoms
def clean_matrix(matrix, eps=1e-12):
    """ clean from small values"""
    matrix = np.array(matrix)
    for ij in np.ndindex(matrix.shape):
        if abs(matrix[ij]) < eps:
            matrix[ij] = 0
    return matrix