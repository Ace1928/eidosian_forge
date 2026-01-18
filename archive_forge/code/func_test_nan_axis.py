import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special import logsumexp
from scipy.stats import circstd
from ...data import from_dict, load_arviz_data
from ...stats.density_utils import histogram
from ...stats.stats_utils import (
from ...stats.stats_utils import logsumexp as _logsumexp
from ...stats.stats_utils import make_ufunc, not_valid, stats_variance_2d, wrap_xarray_ufunc
@pytest.mark.parametrize('axis', (-1, 0, 1))
def test_nan_axis(axis):
    data = np.random.randn(4, 100)
    data[0, 0] = np.nan
    axis_ = len(data.shape) + axis if axis < 0 else axis
    assert not_valid(data, check_shape=False, nan_kwargs=dict(how='any'))
    assert not_valid(data, check_shape=False, nan_kwargs=dict(how='any', axis=axis)).any()
    assert not not_valid(data, check_shape=False, nan_kwargs=dict(how='any', axis=axis)).all()
    assert not_valid(data, check_shape=False, nan_kwargs=dict(how='any', axis=axis)).shape == tuple((dim for ax, dim in enumerate(data.shape) if ax != axis_))