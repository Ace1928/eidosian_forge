import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def evaluate_hessian_equality_constraints(self):
    Th_in = self._input_values[0]
    Th_out = self._input_values[1]
    Tc_in = self._input_values[2]
    Tc_out = self._input_values[3]
    UA = self._input_values[4]
    Q = self._input_values[5]
    lmtd = self._input_values[6]
    dT1 = self._input_values[7]
    dT2 = self._input_values[8]
    row = np.zeros(5, dtype=np.int64)
    col = np.zeros(5, dtype=np.int64)
    data = np.zeros(5, dtype=np.float64)
    lam = self._eq_constraint_multipliers
    idx = 0
    row[idx], col[idx], data[idx] = (7, 6, lam[2] * -1 / dT1)
    idx += 1
    row[idx], col[idx], data[idx] = (7, 7, lam[2] * lmtd / dT1 ** 2)
    idx += 1
    row[idx], col[idx], data[idx] = (8, 6, lam[2] * 1 / dT2)
    idx += 1
    row[idx], col[idx], data[idx] = (8, 8, lam[2] * -lmtd / dT2 ** 2)
    idx += 1
    row[idx], col[idx], data[idx] = (6, 4, lam[3] * 1)
    idx += 1
    assert idx == 5
    return spa.coo_matrix((data, (row, col)), shape=(9, 9))