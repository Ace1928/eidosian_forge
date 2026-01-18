from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
@pytest.mark.parametrize('kind, kwargs', [('skew', {}), ('kurtosis', {}), ('kurtosis', {'fisher': False})])
@pytest.mark.parametrize('single_dim', [True, False])
def test_measures(kind, kwargs, single_dim):
    np.random.seed(seed=1337)
    if single_dim:
        x = np.random.random(size=(30,))
    else:
        x = np.random.random(size=(30, 2))
    y = da.from_array(x, 3)
    dfunc = getattr(dask.array.stats, kind)
    sfunc = getattr(scipy.stats, kind)
    expected = sfunc(x, **kwargs)
    result = dfunc(y, **kwargs)
    if np.isscalar(expected):
        expected = np.array(expected)
    assert_eq(result, expected)
    assert isinstance(result, da.Array)