import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_iir_symmetry(self):
    b, a = gammatone(440, 'iir', fs=24000)
    z, p, k = tf2zpk(b, a)
    assert_array_equal(sorted(z), sorted(z.conj()))
    assert_array_equal(sorted(p), sorted(p.conj()))
    assert_equal(k, np.real(k))
    assert_(issubclass(b.dtype.type, np.floating))
    assert_(issubclass(a.dtype.type, np.floating))