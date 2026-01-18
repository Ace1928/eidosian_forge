import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
@pytest.mark.parametrize('whole,nyquist,worN', [(False, False, 32), (False, True, 32), (True, False, 32), (True, True, 32), (False, False, 257), (False, True, 257), (True, False, 257), (True, True, 257)])
def test_17289(self, whole, nyquist, worN):
    d = [0, 1]
    w, Drfft = freqz(d, worN=32, whole=whole, include_nyquist=nyquist)
    _, Dpoly = freqz(d, worN=w)
    assert_allclose(Drfft, Dpoly)