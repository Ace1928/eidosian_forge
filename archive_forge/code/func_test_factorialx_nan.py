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
@pytest.mark.parametrize('exact', [True, False])
@pytest.mark.parametrize('content', [np.nan, None, np.datetime64('nat')], ids=['NaN', 'None', 'NaT'])
def test_factorialx_nan(self, content, exact):
    assert special.factorial(content, exact=exact) is np.nan
    assert special.factorial2(content, exact=exact) is np.nan
    assert special.factorialk(content, 3, exact=True) is np.nan
    if content is not np.nan:
        with pytest.raises(ValueError, match='Unsupported datatype.*'):
            special.factorial([content], exact=exact)
    elif exact:
        with pytest.warns(DeprecationWarning, match='Non-integer array.*'):
            assert np.isnan(special.factorial([content], exact=exact)[0])
    else:
        assert np.isnan(special.factorial([content], exact=exact)[0])
    with pytest.raises(ValueError, match='factorial2 does not support.*'):
        special.factorial2([content], exact=exact)
    with pytest.raises(ValueError, match='factorialk does not support.*'):
        special.factorialk([content], 3, exact=True)