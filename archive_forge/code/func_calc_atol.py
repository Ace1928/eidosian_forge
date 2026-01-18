import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def calc_atol(h, x0, f, hess, EPS):
    t0 = h / 2 * max(np.abs(hess(x0)), np.abs(hess(x0 + h)))
    t1 = EPS / h * max(np.abs(f(x0)), np.abs(f(x0 + h)))
    return t0 + t1