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
@pytest.mark.parametrize('x, y, z, hue, markersize, row, col, add_legend, add_colorbar', [('A', 'B', None, None, None, None, None, None, None), ('B', 'A', None, 'w', None, None, None, True, None), ('A', 'B', None, 'y', 'x', None, None, True, True), ('A', 'B', 'z', None, None, None, None, None, None), ('B', 'A', 'z', 'w', None, None, None, True, None), ('A', 'B', 'z', 'y', 'x', None, None, True, True), ('A', 'B', 'z', 'y', 'x', 'w', None, True, True)])
def test_datarray_scatter(x, y, z, hue, markersize, row, col, add_legend, add_colorbar) -> None:
    """Test datarray scatter. Merge with TestPlot1D eventually."""
    ds = xr.tutorial.scatter_example_dataset()
    extra_coords = [v for v in [x, hue, markersize] if v is not None]
    coords = dict(ds.coords)
    coords.update({v: ds[v] for v in extra_coords})
    darray = xr.DataArray(ds[y], coords=coords)
    with figure_context():
        darray.plot.scatter(x=x, z=z, hue=hue, markersize=markersize, add_legend=add_legend, add_colorbar=add_colorbar)