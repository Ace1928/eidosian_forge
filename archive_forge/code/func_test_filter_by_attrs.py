from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_filter_by_attrs(self) -> None:
    precip = dict(standard_name='convective_precipitation_flux')
    temp0 = dict(standard_name='air_potential_temperature', height='0 m')
    temp10 = dict(standard_name='air_potential_temperature', height='10 m')
    ds = Dataset({'temperature_0': (['t'], [0], temp0), 'temperature_10': (['t'], [0], temp10), 'precipitation': (['t'], [0], precip)}, coords={'time': (['t'], [0], dict(axis='T', long_name='time_in_seconds'))})
    ds.filter_by_attrs(standard_name='invalid_standard_name')
    new_ds = ds.filter_by_attrs(standard_name='invalid_standard_name')
    assert not bool(new_ds.data_vars)
    new_ds = ds.filter_by_attrs(standard_name='convective_precipitation_flux')
    assert new_ds['precipitation'].standard_name == 'convective_precipitation_flux'
    assert_equal(new_ds['precipitation'], ds['precipitation'])
    new_ds = ds.filter_by_attrs(long_name='time_in_seconds')
    assert new_ds['time'].long_name == 'time_in_seconds'
    assert not bool(new_ds.data_vars)
    new_ds = ds.filter_by_attrs(standard_name='air_potential_temperature')
    assert len(new_ds.data_vars) == 2
    for var in new_ds.data_vars:
        assert new_ds[var].standard_name == 'air_potential_temperature'
    new_ds = ds.filter_by_attrs(height=lambda v: v is not None)
    assert len(new_ds.data_vars) == 2
    for var in new_ds.data_vars:
        assert new_ds[var].standard_name == 'air_potential_temperature'
    new_ds = ds.filter_by_attrs(height='10 m')
    assert len(new_ds.data_vars) == 1
    for var in new_ds.data_vars:
        assert new_ds[var].height == '10 m'
    new_ds = ds.filter_by_attrs(standard_name='convective_precipitation_flux', height='0 m')
    assert not bool(new_ds.data_vars)
    new_ds = ds.filter_by_attrs(standard_name='air_potential_temperature', height='0 m')
    for var in new_ds.data_vars:
        assert new_ds[var].standard_name == 'air_potential_temperature'
        assert new_ds[var].height == '0 m'
        assert new_ds[var].height != '10 m'
    new_ds = ds.filter_by_attrs(standard_name=lambda v: False, height=lambda v: True)
    assert not bool(new_ds.data_vars)