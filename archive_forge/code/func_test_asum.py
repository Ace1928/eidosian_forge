import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_asum(self):
    for p in 'sd':
        f = getattr(fblas, p + 'asum', None)
        if f is None:
            continue
        assert_almost_equal(f([3, -4, 5]), 12)
    for p in ['sc', 'dz']:
        f = getattr(fblas, p + 'asum', None)
        if f is None:
            continue
        assert_almost_equal(f([3j, -4, 3 - 4j]), 14)