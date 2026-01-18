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
def test_infer_line_data(self) -> None:
    current = DataArray(name='I', data=np.array([5, 8]), dims=['t'], coords={'t': (['t'], np.array([0.1, 0.2])), 'V': (['t'], np.array([100, 200]))})
    line = current.plot.line(x='V')[0]
    assert_array_equal(line.get_xdata(), current.coords['V'].values)
    line = current.plot.line()[0]
    assert_array_equal(line.get_xdata(), current.coords['t'].values)