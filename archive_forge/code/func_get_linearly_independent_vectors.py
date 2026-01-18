from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def get_linearly_independent_vectors(vectors_list):
    """

    Args:
        vectors_list:
    """
    independent_vectors_list = []
    for vector in vectors_list:
        if np.any(vector != 0):
            if len(independent_vectors_list) == 0:
                independent_vectors_list.append(np.array(vector))
            elif len(independent_vectors_list) == 1:
                rank = np.linalg.matrix_rank(np.array([independent_vectors_list[0], vector, [0, 0, 0]]))
                if rank == 2:
                    independent_vectors_list.append(np.array(vector))
            elif len(independent_vectors_list) == 2:
                mm = np.array([independent_vectors_list[0], independent_vectors_list[1], vector])
                if np.linalg.det(mm) != 0:
                    independent_vectors_list.append(np.array(vector))
        if len(independent_vectors_list) == 3:
            break
    return independent_vectors_list