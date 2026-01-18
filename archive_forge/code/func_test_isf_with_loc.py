import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
@pytest.mark.parametrize('dist, args', distdiscrete)
def test_isf_with_loc(dist, args):
    try:
        distfn = getattr(stats, dist)
    except TypeError:
        distfn = dist
    np.random.seed(1942349)
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        expected = (_b + loc, _a - 1 + loc)
        res = (distfn.isf(0.0, *args, loc=loc), distfn.isf(1.0, *args, loc=loc))
        npt.assert_array_equal(expected, res)
    re_locs = [np.random.randint(-10, -1, size=(5, 3)), np.zeros((5, 3)), np.random.randint(1, 10, size=(5, 3))]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        expected = (_b + loc, _a - 1 + loc)
        res = (distfn.isf(0.0, *args, loc=loc), distfn.isf(1.0, *args, loc=loc))
        npt.assert_array_equal(expected, res)