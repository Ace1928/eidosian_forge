import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
def symm_iterator():
    for j in range(n):
        for i in range(j, n):
            aij, aji = (a[i][j], a[j][i])
            yield (aij, aji, i == j)