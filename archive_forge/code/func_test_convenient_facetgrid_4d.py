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
@pytest.mark.filterwarnings('ignore:tight_layout cannot')
def test_convenient_facetgrid_4d(self) -> None:
    a = easy_array((10, 15, 2, 3))
    d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
    g = self.plotfunc(d, x='x', y='y', col='columns', row='rows')
    assert_array_equal(g.axs.shape, [3, 2])
    for ax in g.axs.flat:
        assert ax.has_data()