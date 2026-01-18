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
def test2d_1d_2d_coordinates_pcolormesh(self) -> None:
    sz = 10
    y2d, x2d = np.meshgrid(np.arange(sz), np.arange(sz))
    a = DataArray(easy_array((sz, sz)), dims=['x', 'y'], coords={'x2d': (['x', 'y'], x2d), 'y2d': (['x', 'y'], y2d)})
    for x, y in [('x', 'y'), ('y', 'x'), ('x2d', 'y'), ('y', 'x2d'), ('x', 'y2d'), ('y2d', 'x'), ('x2d', 'y2d'), ('y2d', 'x2d')]:
        p = a.plot.pcolormesh(x=x, y=y)
        v = p.get_paths()[0].vertices
        assert isinstance(v, np.ndarray)
        _, unique_counts = np.unique(v[:-1], axis=0, return_counts=True)
        assert np.all(unique_counts == 1)