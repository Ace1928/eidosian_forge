import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('method', methods)
def test_nan_policy_omit_3d(self, axis, method):
    shape = (20, 21, 22)
    rng = np.random.default_rng(602173110498916506)
    a = rng.random(size=shape)
    i = rng.random(size=shape) < 0.4
    j = rng.random(size=shape) < 0.1
    k = rng.random(size=shape) < 0.1
    a[i] = np.nan
    a[j] = -np.inf
    a[k] - np.inf

    def rank_1d_omit(a, method):
        out = np.zeros_like(a)
        i = np.isnan(a)
        a_compressed = a[~i]
        res = rankdata(a_compressed, method)
        out[~i] = res
        out[i] = np.nan
        return out

    def rank_omit(a, method, axis):
        return np.apply_along_axis(lambda a: rank_1d_omit(a, method), axis, a)
    res = rankdata(a, method, axis=axis, nan_policy='omit')
    res0 = rank_omit(a, method, axis=axis)
    assert_array_equal(res, res0)