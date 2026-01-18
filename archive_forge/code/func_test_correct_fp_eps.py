import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_correct_fp_eps():
    EPS = np.finfo(np.float64).eps
    relative_step = {'2-point': EPS ** 0.5, '3-point': EPS ** (1 / 3), 'cs': EPS ** 0.5}
    for method in ['2-point', '3-point', 'cs']:
        assert_allclose(_eps_for_method(np.float64, np.float64, method), relative_step[method])
        assert_allclose(_eps_for_method(np.complex128, np.complex128, method), relative_step[method])
    EPS = np.finfo(np.float32).eps
    relative_step = {'2-point': EPS ** 0.5, '3-point': EPS ** (1 / 3), 'cs': EPS ** 0.5}
    for method in ['2-point', '3-point', 'cs']:
        assert_allclose(_eps_for_method(np.float64, np.float32, method), relative_step[method])
        assert_allclose(_eps_for_method(np.float32, np.float64, method), relative_step[method])
        assert_allclose(_eps_for_method(np.float32, np.float32, method), relative_step[method])