import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_analog_sos(self):
    sos = [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]
    sos2 = iirfilter(N=1, Wn=1, btype='low', analog=True, output='sos')
    assert_array_almost_equal(sos, sos2)