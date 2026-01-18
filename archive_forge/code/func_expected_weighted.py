from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def expected_weighted(da, weights, dim, skipna, operation):
    """
    Generate expected result using ``*`` and ``sum``. This is checked against
    the result of da.weighted which uses ``dot``
    """
    weighted_sum = (da * weights).sum(dim=dim, skipna=skipna)
    if operation == 'sum':
        return weighted_sum
    masked_weights = weights.where(da.notnull())
    sum_of_weights = masked_weights.sum(dim=dim, skipna=True)
    valid_weights = sum_of_weights != 0
    sum_of_weights = sum_of_weights.where(valid_weights)
    if operation == 'sum_of_weights':
        return sum_of_weights
    weighted_mean = weighted_sum / sum_of_weights
    if operation == 'mean':
        return weighted_mean
    demeaned = da - weighted_mean
    sum_of_squares = (demeaned ** 2 * weights).sum(dim=dim, skipna=skipna)
    if operation == 'sum_of_squares':
        return sum_of_squares
    var = sum_of_squares / sum_of_weights
    if operation == 'var':
        return var
    if operation == 'std':
        return np.sqrt(var)