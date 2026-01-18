import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_identity_filter(self):
    w, gd = group_delay((1, 1))
    assert_array_almost_equal(w, pi * np.arange(512) / 512)
    assert_array_almost_equal(gd, np.zeros(512))
    w, gd = group_delay((1, 1), whole=True)
    assert_array_almost_equal(w, 2 * pi * np.arange(512) / 512)
    assert_array_almost_equal(gd, np.zeros(512))