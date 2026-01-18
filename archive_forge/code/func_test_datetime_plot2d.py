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
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_datetime_plot2d(self) -> None:
    da = DataArray(np.arange(3 * 4).reshape(3, 4), dims=('x', 'y'), coords={'x': [1, 2, 3], 'y': [np.datetime64(f'2000-01-{x:02d}') for x in range(1, 5)]})
    p = da.plot.pcolormesh()
    ax = p.axes
    assert ax is not None
    assert type(ax.xaxis.get_major_locator()) is mpl.dates.AutoDateLocator