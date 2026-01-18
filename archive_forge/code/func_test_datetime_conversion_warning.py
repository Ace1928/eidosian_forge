from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@pytest.mark.parametrize(('values', 'warns_under_pandas_version_two'), [(np.datetime64('2000-01-01', 'ns'), False), (np.datetime64('2000-01-01', 's'), True), (np.array([np.datetime64('2000-01-01', 'ns')]), False), (np.array([np.datetime64('2000-01-01', 's')]), True), (pd.date_range('2000', periods=1), False), (datetime(2000, 1, 1), False), (np.array([datetime(2000, 1, 1)]), False), (pd.date_range('2000', periods=1, tz=pytz.timezone('America/New_York')), False), (pd.Series(pd.date_range('2000', periods=1, tz=pytz.timezone('America/New_York'))), False)], ids=lambda x: f'{x}')
def test_datetime_conversion_warning(values, warns_under_pandas_version_two) -> None:
    dims = ['time'] if isinstance(values, (np.ndarray, pd.Index, pd.Series)) else []
    if warns_under_pandas_version_two and has_pandas_version_two:
        with pytest.warns(UserWarning, match='non-nanosecond precision datetime'):
            var = Variable(dims, values)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            var = Variable(dims, values)
    if var.dtype.kind == 'M':
        assert var.dtype == np.dtype('datetime64[ns]')
    else:
        assert isinstance(var._data, PandasIndexingAdapter)
        assert var._data.array.dtype == pd.DatetimeTZDtype('ns', pytz.timezone('America/New_York'))