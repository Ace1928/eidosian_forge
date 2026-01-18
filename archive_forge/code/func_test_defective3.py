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
def test_defective3(self):
    a = array([[-2.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -3.0, 10.0, 3.0, 3.0, 3.0, 0.0], [0.0, 0.0, 2.0, 15.0, 3.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 15.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 25.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0]])
    signm(a, disp=False)