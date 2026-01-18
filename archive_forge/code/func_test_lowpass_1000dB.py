import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_lowpass_1000dB(self):
    wp = 0.2
    ws = 0.3
    rp = 3
    rs = 1000
    N, Wn = ellipord(wp, ws, rp, rs, False)
    sos = ellip(N, rp, rs, Wn, 'lp', False, output='sos')
    w, h = sosfreqz(sos)
    w /= np.pi
    assert_array_less(-rp - 0.1, dB(h[w <= wp]))
    assert_array_less(dB(h[ws <= w]), -rs + 0.1)