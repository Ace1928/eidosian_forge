import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def fun_vector_vector(self, x):
    return np.array([x[0] * np.sin(x[1]), x[1] * np.cos(x[0]), x[0] ** 3 * x[1] ** (-0.5)])