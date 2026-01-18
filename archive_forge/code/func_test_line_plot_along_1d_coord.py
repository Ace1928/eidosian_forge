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
def test_line_plot_along_1d_coord(self) -> None:
    x_coord = xr.DataArray(data=[0.1, 0.2], dims=['x'])
    t_coord = xr.DataArray(data=[10, 20], dims=['t'])
    da = xr.DataArray(data=np.array([[0, 1], [5, 9]]), dims=['x', 't'], coords={'x': x_coord, 'time': t_coord})
    line = da.plot(x='time', hue='x')[0]
    assert_array_equal(line.get_xdata(), da.coords['time'].values)
    line = da.plot(y='time', hue='x')[0]
    assert_array_equal(line.get_ydata(), da.coords['time'].values)