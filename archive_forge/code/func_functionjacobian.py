import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def functionjacobian(x):
    return np.array([2.0 * x[0], 2.0 * x[1]])