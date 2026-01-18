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
def test_coord_with_interval_step_x_and_y_raises_valueeerror(self) -> None:
    """Test that step plot with intervals both on x and y axes raises an error."""
    arr = xr.DataArray([pd.Interval(0, 1), pd.Interval(1, 2)], coords=[('x', [pd.Interval(0, 1), pd.Interval(1, 2)])])
    with pytest.raises(TypeError, match='intervals against intervals'):
        arr.plot.step()