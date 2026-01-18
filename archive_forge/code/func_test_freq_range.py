import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_freq_range(self):
    z = []
    p = [-1]
    k = 1
    n = 10
    expected_w = np.logspace(-2, 1, n)
    w, H = freqs_zpk(z, p, k, worN=n)
    assert_array_almost_equal(w, expected_w)