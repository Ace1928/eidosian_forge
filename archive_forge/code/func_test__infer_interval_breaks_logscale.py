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
def test__infer_interval_breaks_logscale(self) -> None:
    """
        Check if interval breaks are defined in the logspace if scale="log"
        """
    x = np.logspace(-4, 3, 8)
    expected_interval_breaks = 10 ** np.linspace(-4.5, 3.5, 9)
    np.testing.assert_allclose(_infer_interval_breaks(x, scale='log'), expected_interval_breaks)
    x = np.logspace(-4, 3, 8)
    y = np.linspace(-5, 5, 11)
    x, y = np.meshgrid(x, y)
    expected_interval_breaks = np.vstack([10 ** np.linspace(-4.5, 3.5, 9)] * 12)
    x = _infer_interval_breaks(x, axis=1, scale='log')
    x = _infer_interval_breaks(x, axis=0, scale='log')
    np.testing.assert_allclose(x, expected_interval_breaks)