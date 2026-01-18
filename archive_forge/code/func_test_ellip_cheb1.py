import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_ellip_cheb1(self):
    n, wn = cheb1ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    assert n == 7
    n2, w2 = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    assert not (wn == w2).all()