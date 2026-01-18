import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
import numpy as np
from scipy.sparse import coo_matrix, tril
from pyomo.contrib import interior_point as ip
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def get_base_matrix_wrong_order(use_tril):
    if use_tril:
        row = [1, 0, 1, 2, 2]
        col = [0, 0, 1, 0, 2]
        data = [7, 1, 4, 3, 6]
    else:
        row = [1, 0, 0, 0, 1, 2, 2]
        col = [0, 1, 2, 0, 1, 0, 2]
        data = [7, 7, 3, 1, 4, 3, 6]
    mat = coo_matrix((data, (row, col)), shape=(3, 3), dtype=np.double)
    return mat