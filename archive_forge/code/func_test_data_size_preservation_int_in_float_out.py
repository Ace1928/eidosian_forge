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
def test_data_size_preservation_int_in_float_out(self):
    M = np.zeros((10, 10), dtype=np.int8)
    assert sqrtm(M).dtype == np.float16
    M = np.zeros((10, 10), dtype=np.int16)
    assert sqrtm(M).dtype == np.float16
    M = np.zeros((10, 10), dtype=np.int32)
    assert sqrtm(M).dtype == np.float32
    M = np.zeros((10, 10), dtype=np.int64)
    assert sqrtm(M).dtype == np.float64