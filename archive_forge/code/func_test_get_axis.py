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
@pytest.mark.parametrize(['figsize', 'size', 'aspect', 'ax', 'kwargs'], [pytest.param((3, 2), None, None, False, {}, id='figsize'), pytest.param((3.5, 2.5), None, None, False, {'label': 'test'}, id='figsize_kwargs'), pytest.param(None, 5, None, False, {}, id='size'), pytest.param(None, 5.5, None, False, {'label': 'test'}, id='size_kwargs'), pytest.param(None, 5, 1, False, {}, id='size+aspect'), pytest.param(None, 5, 'auto', False, {}, id='auto_aspect'), pytest.param(None, 5, 'equal', False, {}, id='equal_aspect'), pytest.param(None, None, None, True, {}, id='ax'), pytest.param(None, None, None, False, {}, id='default'), pytest.param(None, None, None, False, {'label': 'test'}, id='default_kwargs')])
def test_get_axis(figsize: tuple[float, float] | None, size: float | None, aspect: float | None, ax: bool, kwargs: dict[str, Any]) -> None:
    with figure_context():
        inp_ax = plt.axes() if ax else None
        out_ax = get_axis(figsize=figsize, size=size, aspect=aspect, ax=inp_ax, **kwargs)
        assert isinstance(out_ax, mpl.axes.Axes)