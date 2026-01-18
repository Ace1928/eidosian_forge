from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
@requires_matplotlib
def test_assert_valid_xy() -> None:
    ds = xr.tutorial.scatter_example_dataset()
    darray = ds.A
    _assert_valid_xy(darray=darray, xy='x', name='x')
    _assert_valid_xy(darray=darray, xy=None, name='x')
    with pytest.raises(ValueError, match='x must be one of'):
        _assert_valid_xy(darray=darray, xy='error_now', name='x')