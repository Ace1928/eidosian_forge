import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_nearest_divisor(self):
    b, a = iircomb(50 / int(44100 / 2), 50.0, ftype='notch')
    freqs, response = freqz(b, a, [22000], fs=44100)
    assert abs(response[0]) < 1e-10