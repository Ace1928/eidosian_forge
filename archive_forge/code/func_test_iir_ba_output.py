import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_iir_ba_output(self):
    b, a = gammatone(440, 'iir', fs=16000)
    b2 = [1.31494461367464e-06, -5.03391196645395e-06, 7.00649426000897e-06, -4.18951968419854e-06, 9.02614910412011e-07]
    a2 = [1.0, -7.65646235454218, 25.7584699322366, -49.7319214483238, 60.2667361289181, -46.9399590980486, 22.9474798808461, -6.43799381299034, 0.793651554625368]
    assert_allclose(b, b2)
    assert_allclose(a, a2)