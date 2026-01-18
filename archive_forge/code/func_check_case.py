import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def check_case(self, n, sym, low):
    assert_array_equal(pascal(n), sym)
    assert_array_equal(pascal(n, kind='lower'), low)
    assert_array_equal(pascal(n, kind='upper'), low.T)
    assert_array_almost_equal(pascal(n, exact=False), sym)
    assert_array_almost_equal(pascal(n, exact=False, kind='lower'), low)
    assert_array_almost_equal(pascal(n, exact=False, kind='upper'), low.T)