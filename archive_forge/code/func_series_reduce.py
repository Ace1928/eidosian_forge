from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
def series_reduce(da, func, dim, **kwargs):
    """convert DataArray to pd.Series, apply pd.func, then convert back to
    a DataArray. Multiple dims cannot be specified."""
    if kwargs.get('skipna', True) is None:
        kwargs['skipna'] = True
    if dim is None or da.ndim == 1:
        se = da.to_series()
        return from_series_or_scalar(getattr(se, func)(**kwargs))
    else:
        da1 = []
        dims = list(da.dims)
        dims.remove(dim)
        d = dims[0]
        for i in range(len(da[d])):
            da1.append(series_reduce(da.isel(**{d: i}), func, dim, **kwargs))
        if d in da.coords:
            return concat(da1, dim=da[d])
        return concat(da1, dim=d)