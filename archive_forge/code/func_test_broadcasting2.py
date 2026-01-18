import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_broadcasting2(self):
    np.random.seed(123)
    b = np.random.rand(3, 5, 1)
    for whole in [False, True]:
        for worN in [16, 17, np.linspace(0, 1, 10)]:
            w, h = freqz(b, worN=worN, whole=whole)
            for k in range(b.shape[1]):
                bk = b[:, k, 0]
                ww, hh = freqz(bk, worN=worN, whole=whole)
                assert_allclose(ww, w)
                assert_allclose(hh, h[k])