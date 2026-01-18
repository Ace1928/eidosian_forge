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
def test_geo_data(self) -> None:
    lat = np.array([[16.28, 18.48, 19.58, 19.54, 18.35], [28.07, 30.52, 31.73, 31.68, 30.37], [39.65, 42.27, 43.56, 43.51, 42.11], [50.52, 53.22, 54.55, 54.5, 53.06]])
    lon = np.array([[-126.13, -113.69, -100.92, -88.04, -75.29], [-129.27, -115.62, -101.54, -87.32, -73.26], [-133.1, -118.0, -102.31, -86.42, -70.76], [-137.85, -120.99, -103.28, -85.28, -67.62]])
    data = np.sqrt(lon ** 2 + lat ** 2)
    da = DataArray(data, dims=('y', 'x'), coords={'lon': (('y', 'x'), lon), 'lat': (('y', 'x'), lat)})
    da.plot(x='lon', y='lat')
    ax = plt.gca()
    assert ax.has_data()
    da.plot(x='lat', y='lon')
    ax = plt.gca()
    assert ax.has_data()