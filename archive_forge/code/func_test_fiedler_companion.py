import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def test_fiedler_companion():
    fc = fiedler_companion([])
    assert_equal(fc.size, 0)
    fc = fiedler_companion([1.0])
    assert_equal(fc.size, 0)
    fc = fiedler_companion([1.0, 2.0])
    assert_array_equal(fc, np.array([[-2.0]]))
    fc = fiedler_companion([1e-12, 2.0, 3.0])
    assert_array_almost_equal(fc, companion([1e-12, 2.0, 3.0]))
    with assert_raises(ValueError):
        fiedler_companion([0, 1, 2])
    fc = fiedler_companion([1.0, -16.0, 86.0, -176.0, 105.0])
    assert_array_almost_equal(eigvals(fc), np.array([7.0, 5.0, 3.0, 1.0]))