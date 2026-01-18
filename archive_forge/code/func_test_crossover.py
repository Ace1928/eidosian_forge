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
def test_crossover(self):
    A_eq, b_eq, c, _, _ = magic_square(4)
    bounds = (0, 1)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, options=self.options)
    assert_equal(res.crossover_nit == 0, self.method != 'highs-ipm')