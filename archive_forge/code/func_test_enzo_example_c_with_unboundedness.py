import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def test_enzo_example_c_with_unboundedness(self):
    m = 50
    c = -np.ones(m)
    tmp = 2 * np.pi * np.arange(m) / (m + 1)
    row0 = np.cos(tmp) - 1
    row0[0] = 0.0
    row1 = np.sin(tmp)
    row1[0] = 0.0
    A_eq = np.vstack((row0, row1))
    b_eq = [0, 0]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_unbounded(res)