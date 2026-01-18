import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def test_fiedler():
    f = fiedler([])
    assert_equal(f.size, 0)
    f = fiedler([123.0])
    assert_array_equal(f, np.array([[0.0]]))
    f = fiedler(np.arange(1, 7))
    des = np.array([[0, 1, 2, 3, 4, 5], [1, 0, 1, 2, 3, 4], [2, 1, 0, 1, 2, 3], [3, 2, 1, 0, 1, 2], [4, 3, 2, 1, 0, 1], [5, 4, 3, 2, 1, 0]])
    assert_array_equal(f, des)