import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
@pytest.mark.parametrize('z, s, c', [(0.5, 0.06473243285999929, 0.49234422587144644), (0.5 + 0j, 0.06473243285999929, 0.49234422587144644), (-2.0 + 0.1j, -0.3109538687728942 - 0.0005870728836383176j, -0.4879956866358554 + 0.10670801832903172j), (-0.1 - 1.5j, -0.03918309471866977 + 0.7197508454568574j, 0.09605692502968956 - 0.43625191013617465j), (6.0, 0.44696076, 0.49953147), (6.0 + 0j, 0.44696076, 0.49953147), (6j, -0.44696076j, 0.49953147j), (-6.0 + 0j, -0.44696076, -0.49953147), (-6j, 0.44696076j, -0.49953147j), (np.inf, 0.5, 0.5), (-np.inf, -0.5, -0.5)])
def test_fresnel_values(self, z, s, c):
    frs = array(special.fresnel(z))
    assert_array_almost_equal(frs, array([s, c]), 8)