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
def test_nan_inf_are_ignored(self) -> None:
    cmap_params1 = _determine_cmap_params(self.data)
    data = self.data
    data[50:55] = np.nan
    data[56:60] = np.inf
    cmap_params2 = _determine_cmap_params(data)
    assert cmap_params1['vmin'] == cmap_params2['vmin']
    assert cmap_params1['vmax'] == cmap_params2['vmax']