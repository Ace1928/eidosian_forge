import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_invalid_wn_range(self):
    assert_raises(ValueError, iirfilter, 1, 2, btype='low')
    assert_raises(ValueError, iirfilter, 1, [0.5, 1], btype='band')
    assert_raises(ValueError, iirfilter, 1, [0.0, 0.5], btype='band')
    assert_raises(ValueError, iirfilter, 1, -1, btype='high')
    assert_raises(ValueError, iirfilter, 1, [1, 2], btype='band')
    assert_raises(ValueError, iirfilter, 1, [10, 20], btype='stop')
    with pytest.raises(ValueError, match='must be greater than 0'):
        iirfilter(2, 0, btype='low', analog=True)
    with pytest.raises(ValueError, match='must be greater than 0'):
        iirfilter(2, -1, btype='low', analog=True)
    with pytest.raises(ValueError, match='must be greater than 0'):
        iirfilter(2, [0, 100], analog=True)
    with pytest.raises(ValueError, match='must be greater than 0'):
        iirfilter(2, [-1, 100], analog=True)
    with pytest.raises(ValueError, match='must be greater than 0'):
        iirfilter(2, [10, 0], analog=True)
    with pytest.raises(ValueError, match='must be greater than 0'):
        iirfilter(2, [10, -1], analog=True)