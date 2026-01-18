from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
def test_kurtosis_single_return_type():
    """This function tests the return type for the kurtosis method for a 1d array."""
    numpy_array = np.random.random(size=(30,))
    dask_array = da.from_array(numpy_array, 3)
    result = dask.array.stats.kurtosis(dask_array).compute()
    result_non_fisher = dask.array.stats.kurtosis(dask_array, fisher=False).compute()
    assert isinstance(result, np.float64)
    assert isinstance(result_non_fisher, np.float64)