import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def jac_parametrized(self, x, c0, c1=0.1):
    return np.array([[c0 * np.exp(c0 * x[0]), 0], [0, c1 * np.exp(c1 * x[1])]])