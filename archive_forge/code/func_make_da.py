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
def make_da():
    da = xr.DataArray(np.ones((10, 20)), dims=['x', 'y'], coords={'x': np.arange(10), 'y': np.arange(100, 120)}, name='a').chunk({'x': 4, 'y': 5})
    da.x.attrs['long_name'] = 'x'
    da.attrs['test'] = 'test'
    da.coords['c2'] = 0.5
    da.coords['ndcoord'] = da.x * 2
    da.coords['cxy'] = (da.x * da.y).chunk({'x': 4, 'y': 5})
    return da