from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@requires_flox
def test_default_flox_method() -> None:
    import flox.xarray
    da = xr.DataArray([1, 2, 3], dims='x', coords={'label': ('x', [2, 2, 1])})
    result = xr.DataArray([3, 3], dims='label', coords={'label': [1, 2]})
    with mock.patch('flox.xarray.xarray_reduce', return_value=result) as mocked_reduce:
        da.groupby('label').sum()
    kwargs = mocked_reduce.call_args.kwargs
    if Version(flox.__version__) < Version('0.9.0'):
        assert kwargs['method'] == 'cohorts'
    else:
        assert 'method' not in kwargs