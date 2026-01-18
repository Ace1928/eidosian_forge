import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_ger(self):
    for p in 'sd':
        f = getattr(fblas, p + 'ger', None)
        if f is None:
            continue
        assert_array_almost_equal(f(1, [1, 2], [3, 4]), [[3, 4], [6, 8]])
        assert_array_almost_equal(f(2, [1, 2, 3], [3, 4]), [[6, 8], [12, 16], [18, 24]])
        assert_array_almost_equal(f(1, [1, 2], [3, 4], a=[[1, 2], [3, 4]]), [[4, 6], [9, 12]])
    for p in 'cz':
        f = getattr(fblas, p + 'geru', None)
        if f is None:
            continue
        assert_array_almost_equal(f(1, [1j, 2], [3, 4]), [[3j, 4j], [6, 8]])
        assert_array_almost_equal(f(-2, [1j, 2j, 3j], [3j, 4j]), [[6, 8], [12, 16], [18, 24]])
    for p in 'cz':
        for name in ('ger', 'gerc'):
            f = getattr(fblas, p + name, None)
            if f is None:
                continue
            assert_array_almost_equal(f(1, [1j, 2], [3, 4]), [[3j, 4j], [6, 8]])
            assert_array_almost_equal(f(2, [1j, 2j, 3j], [3j, 4j]), [[6, 8], [12, 16], [18, 24]])