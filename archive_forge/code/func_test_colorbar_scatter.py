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
def test_colorbar_scatter(self) -> None:
    ds = Dataset({'a': (('x', 'y'), np.arange(4).reshape(2, 2))})
    fg: xplt.FacetGrid = ds.plot.scatter(x='a', y='a', row='x', hue='a')
    cbar = fg.cbar
    assert cbar is not None
    assert hasattr(cbar, 'vmin')
    assert cbar.vmin == 0
    assert hasattr(cbar, 'vmax')
    assert cbar.vmax == 3