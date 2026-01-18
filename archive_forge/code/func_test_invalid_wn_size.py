import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_invalid_wn_size(self):
    assert_raises(ValueError, iirfilter, 1, [0.1, 0.9], btype='low')
    assert_raises(ValueError, iirfilter, 1, [0.2, 0.5], btype='high')
    assert_raises(ValueError, iirfilter, 1, 0.2, btype='bp')
    assert_raises(ValueError, iirfilter, 1, 400, btype='bs', analog=True)