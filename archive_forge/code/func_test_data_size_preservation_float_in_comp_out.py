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
def test_data_size_preservation_float_in_comp_out(self):
    M = np.array([[2, 4], [0, -2]], dtype=np.float16)
    assert sqrtm(M).dtype == np.complex64
    M = np.array([[2, 4], [0, -2]], dtype=np.float32)
    assert sqrtm(M).dtype == np.complex64
    M = np.array([[2, 4], [0, -2]], dtype=np.float64)
    assert sqrtm(M).dtype == np.complex128
    if hasattr(np, 'float128') and hasattr(np, 'complex256'):
        M = np.array([[2, 4], [0, -2]], dtype=np.float128)
        assert sqrtm(M).dtype == np.complex256