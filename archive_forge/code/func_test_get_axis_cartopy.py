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
@requires_cartopy
@pytest.mark.parametrize(['figsize', 'size', 'aspect'], [pytest.param((3, 2), None, None, id='figsize'), pytest.param(None, 5, None, id='size'), pytest.param(None, 5, 1, id='size+aspect'), pytest.param(None, None, None, id='default')])
def test_get_axis_cartopy(figsize: tuple[float, float] | None, size: float | None, aspect: float | None) -> None:
    kwargs = {'projection': cartopy.crs.PlateCarree()}
    with figure_context():
        out_ax = get_axis(figsize=figsize, size=size, aspect=aspect, **kwargs)
        assert isinstance(out_ax, cartopy.mpl.geoaxes.GeoAxesSubplot)