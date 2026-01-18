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
def resample_as_pandas(array, *args, **kwargs):
    array_ = array.copy(deep=True)
    if use_cftime:
        array_['time'] = times.to_datetimeindex()
    result = DataArray.from_series(array_.to_series().resample(*args, **kwargs).mean())
    if use_cftime:
        result = result.convert_calendar(calendar='standard', use_cftime=use_cftime)
    return result