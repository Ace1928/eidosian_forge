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
def test_get_axis_raises():
    with pytest.raises(ValueError, match='both `figsize` and `ax`'):
        get_axis(figsize=[4, 4], size=None, aspect=None, ax='something')
    with pytest.raises(ValueError, match='both `size` and `ax`'):
        get_axis(figsize=None, size=200, aspect=4 / 3, ax='something')
    with pytest.raises(ValueError, match='both `figsize` and `size`'):
        get_axis(figsize=[4, 4], size=200, aspect=None, ax=None)
    with pytest.raises(ValueError, match='`aspect` argument without `size`'):
        get_axis(figsize=None, size=None, aspect=4 / 3, ax=None)
    with pytest.raises(ValueError, match='cannot use subplot_kws with existing ax'):
        get_axis(figsize=None, size=None, aspect=None, ax=1, something_else=5)