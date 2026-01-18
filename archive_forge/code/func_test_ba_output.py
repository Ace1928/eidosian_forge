import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_ba_output(self):
    b_notch, a_notch = iircomb(60, 35, ftype='notch', fs=600)
    b_notch2 = [0.957020174408697, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.957020174408697]
    a_notch2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.914040348817395]
    assert_allclose(b_notch, b_notch2)
    assert_allclose(a_notch, a_notch2)
    b_peak, a_peak = iircomb(60, 35, ftype='peak', fs=600)
    b_peak2 = [0.0429798255913026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0429798255913026]
    a_peak2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.914040348817395]
    assert_allclose(b_peak, b_peak2)
    assert_allclose(a_peak, a_peak2)