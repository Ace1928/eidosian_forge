from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_dataset(ds):
    actual = ds.interpolate_na(dim='time')
    assert actual['var1'].count('time') == actual.sizes['time']
    assert_array_equal(actual['var2'], ds['var2'])