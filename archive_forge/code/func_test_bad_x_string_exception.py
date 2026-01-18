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
def test_bad_x_string_exception(self) -> None:
    with pytest.raises(ValueError, match='x and y cannot be equal.'):
        self.plotmethod(x='y', y='y')
    error_msg = "must be one of None, 'x', 'x2d', 'y', 'y2d'"
    with pytest.raises(ValueError, match=f'x {error_msg}'):
        self.plotmethod(x='not_a_real_dim', y='y')
    with pytest.raises(ValueError, match=f'x {error_msg}'):
        self.plotmethod(x='not_a_real_dim')
    with pytest.raises(ValueError, match=f'y {error_msg}'):
        self.plotmethod(y='not_a_real_dim')
    self.darray.coords['z'] = 100