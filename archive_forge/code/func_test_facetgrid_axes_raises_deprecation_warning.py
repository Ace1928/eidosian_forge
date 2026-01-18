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
def test_facetgrid_axes_raises_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match='self.axes is deprecated since 2022.11 in order to align with matplotlibs plt.subplots, use self.axs instead.'):
        with figure_context():
            ds = xr.tutorial.scatter_example_dataset()
            g = ds.plot.scatter(x='A', y='B', col='x')
            g.axes