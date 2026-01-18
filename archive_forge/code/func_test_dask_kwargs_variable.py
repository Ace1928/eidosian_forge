from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
@pytest.mark.parametrize('method', ['load', 'compute'])
def test_dask_kwargs_variable(method):
    chunked_array = da.from_array(np.arange(3), chunks=(2,))
    x = Variable('y', chunked_array)
    with mock.patch.object(da, 'compute', return_value=(np.arange(3),)) as mock_compute:
        getattr(x, method)(foo='bar')
    mock_compute.assert_called_with(chunked_array, foo='bar')