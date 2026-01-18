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
def test_maybe_gca() -> None:
    with figure_context():
        ax = _maybe_gca(aspect=1)
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_aspect() == 1
    with figure_context():
        plt.figure()
        ax = _maybe_gca(aspect=1)
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_aspect() == 1
    with figure_context():
        existing_axes = plt.axes()
        ax = _maybe_gca(aspect=1)
        assert existing_axes == ax
        assert ax.get_aspect() == 'auto'