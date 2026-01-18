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
def test__infer_interval_breaks(self) -> None:
    assert_array_equal([-0.5, 0.5, 1.5], _infer_interval_breaks([0, 1]))
    assert_array_equal([-0.5, 0.5, 5.0, 9.5, 10.5], _infer_interval_breaks([0, 1, 9, 10]))
    assert_array_equal(pd.date_range('20000101', periods=4) - np.timedelta64(12, 'h'), _infer_interval_breaks(pd.date_range('20000101', periods=3)))
    xref, yref = np.meshgrid(np.arange(6), np.arange(5))
    cx = (xref[1:, 1:] + xref[:-1, :-1]) / 2
    cy = (yref[1:, 1:] + yref[:-1, :-1]) / 2
    x = _infer_interval_breaks(cx, axis=1)
    x = _infer_interval_breaks(x, axis=0)
    y = _infer_interval_breaks(cy, axis=1)
    y = _infer_interval_breaks(y, axis=0)
    np.testing.assert_allclose(xref, x)
    np.testing.assert_allclose(yref, y)
    with pytest.raises(ValueError):
        _infer_interval_breaks(np.array([0, 2, 1]), check_monotonic=True)