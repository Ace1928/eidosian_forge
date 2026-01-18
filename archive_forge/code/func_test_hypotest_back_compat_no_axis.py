from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
@pytest.mark.parametrize(('fun', 'nsamp'), [(stats.kstat, 1), (stats.kstatvar, 1)])
def test_hypotest_back_compat_no_axis(fun, nsamp):
    m, n = (8, 9)
    rng = np.random.default_rng(0)
    x = rng.random((nsamp, m, n))
    res = fun(*x)
    res2 = fun(*x, _no_deco=True)
    res3 = fun([xi.ravel() for xi in x])
    assert_equal(res, res2)
    assert_equal(res, res3)