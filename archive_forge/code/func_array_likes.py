import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture(params=['memoryview', 'array', pytest.param('dask', marks=td.skip_if_no('dask.array')), pytest.param('xarray', marks=td.skip_if_no('xarray'))])
def array_likes(request):
    """
    Fixture giving a numpy array and a parametrized 'data' object, which can
    be a memoryview, array, dask or xarray object created from the numpy array.
    """
    arr = np.array([1, 2, 3], dtype=np.int64)
    name = request.param
    if name == 'memoryview':
        data = memoryview(arr)
    elif name == 'array':
        data = array.array('i', arr)
    elif name == 'dask':
        import dask.array
        data = dask.array.array(arr)
    elif name == 'xarray':
        import xarray as xr
        data = xr.DataArray(arr)
    return (arr, data)