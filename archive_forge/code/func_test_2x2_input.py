import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_2x2_input(self):
    E = np.e
    a = array([[1, 4], [1, 1]])
    aa = (E ** 4 + 1) / (2 * E)
    bb = (E ** 4 - 1) / E
    assert_allclose(expm(a), array([[aa, bb], [bb / 4, aa]]))
    assert expm(a.astype(np.complex64)).dtype.char == 'F'
    assert expm(a.astype(np.float32)).dtype.char == 'f'