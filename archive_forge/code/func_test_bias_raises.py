from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
def test_bias_raises():
    x = np.random.random(size=(30, 2))
    y = da.from_array(x, 3)
    with pytest.raises(NotImplementedError):
        dask.array.stats.skew(y, bias=False)
    with pytest.raises(NotImplementedError):
        dask.array.stats.kurtosis(y, bias=False)