import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def jac_vector_vector(self, x):
    return np.array([[np.sin(x[1]), x[0] * np.cos(x[1])], [-x[1] * np.sin(x[0]), np.cos(x[0])], [3 * x[0] ** 2 * x[1] ** (-0.5), -0.5 * x[0] ** 3 * x[1] ** (-1.5)]])