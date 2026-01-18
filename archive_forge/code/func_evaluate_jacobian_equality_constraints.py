import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def evaluate_jacobian_equality_constraints(self):
    Th_in = self._input_values[0]
    Th_out = self._input_values[1]
    Tc_in = self._input_values[2]
    Tc_out = self._input_values[3]
    UA = self._input_values[4]
    Q = self._input_values[5]
    lmtd = self._input_values[6]
    dT1 = self._input_values[7]
    dT2 = self._input_values[8]
    row = np.zeros(18, dtype=np.int64)
    col = np.zeros(18, dtype=np.int64)
    data = np.zeros(18, dtype=np.float64)
    idx = 0
    row[idx], col[idx], data[idx] = (0, 0, 1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (0, 3, -1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (0, 7, -1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (1, 1, 1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (1, 2, -1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (1, 8, -1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (2, 6, math.log(dT2 / dT1))
    idx += 1
    row[idx], col[idx], data[idx] = (2, 7, -lmtd / dT1 + 1)
    idx += 1
    row[idx], col[idx], data[idx] = (2, 8, lmtd / dT2 - 1)
    idx += 1
    row[idx], col[idx], data[idx] = (3, 4, lmtd)
    idx += 1
    row[idx], col[idx], data[idx] = (3, 5, -1.0)
    idx += 1
    row[idx], col[idx], data[idx] = (3, 6, UA)
    idx += 1
    row[idx], col[idx], data[idx] = (4, 0, self._Fh * self._Cp_h)
    idx += 1
    row[idx], col[idx], data[idx] = (4, 1, -self._Fh * self._Cp_h)
    idx += 1
    row[idx], col[idx], data[idx] = (4, 5, -1)
    idx += 1
    row[idx], col[idx], data[idx] = (5, 2, -self._Fc * self._Cp_c)
    idx += 1
    row[idx], col[idx], data[idx] = (5, 3, self._Fc * self._Cp_c)
    idx += 1
    row[idx], col[idx], data[idx] = (5, 5, -1.0)
    idx += 1
    assert idx == 18
    return spa.coo_matrix((data, (row, col)), shape=(6, 9))