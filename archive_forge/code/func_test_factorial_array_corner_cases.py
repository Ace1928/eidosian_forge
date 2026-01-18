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
@pytest.mark.parametrize('dtype', [np.int64, np.float64, np.complex128, object])
@pytest.mark.parametrize('exact', [True, False])
@pytest.mark.parametrize('dim', range(0, 5))
@pytest.mark.parametrize('content', [[], [1], [1.1], [np.nan], [np.nan, 1]], ids=['[]', '[1]', '[1.1]', '[NaN]', '[NaN, 1]'])
def test_factorial_array_corner_cases(self, content, dim, exact, dtype):
    if dtype == np.int64 and any((np.isnan(x) for x in content)):
        pytest.skip('impossible combination')
    content = content if dim > 0 or len(content) != 1 else content[0]
    n = np.array(content, ndmin=dim, dtype=dtype)
    result = None
    if not content:
        result = special.factorial(n, exact=exact)
    elif not (np.issubdtype(n.dtype, np.integer) or np.issubdtype(n.dtype, np.floating)):
        with pytest.raises(ValueError, match='Unsupported datatype*'):
            special.factorial(n, exact=exact)
    elif exact and (not np.issubdtype(n.dtype, np.integer)) and n.size and np.allclose(n[~np.isnan(n)], n[~np.isnan(n)].astype(np.int64)):
        with pytest.warns(DeprecationWarning, match='Non-integer array.*'):
            result = special.factorial(n, exact=exact)
            if np.any(np.isnan(n)):
                dtype = np.dtype(np.float64)
            else:
                dtype = np.dtype(int)
    elif exact and (not np.issubdtype(n.dtype, np.integer)):
        with pytest.raises(ValueError, match='factorial with exact=.*'):
            special.factorial(n, exact=exact)
    else:
        result = special.factorial(n, exact=exact)

    def assert_really_equal(x, y):
        assert type(x) == type(y), f'types not equal: {type(x)}, {type(y)}'
        assert_equal(x, y)
    if result is not None:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            n_flat = n.ravel() if n.ndim else n
            r = special.factorial(n_flat, exact=exact) if n.size else []
        expected = np.array(r, ndmin=dim, dtype=dtype)
        assert_really_equal(result, expected)