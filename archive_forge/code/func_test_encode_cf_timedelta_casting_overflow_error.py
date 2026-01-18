from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize('use_dask', [False, pytest.param(True, marks=requires_dask)])
@pytest.mark.parametrize('dtype', [np.dtype('int16'), np.dtype('float16')])
def test_encode_cf_timedelta_casting_overflow_error(use_dask, dtype) -> None:
    timedeltas = pd.timedelta_range(start='0h', freq='5h', periods=3)
    encoding = dict(units='microseconds', dtype=dtype)
    variable = Variable(['time'], timedeltas, encoding=encoding)
    if use_dask:
        variable = variable.chunk({'time': 1})
    with pytest.raises(OverflowError, match='Not possible'):
        encoded = conventions.encode_cf_variable(variable)
        encoded.compute()