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
@pytest.mark.slow
def test_figure_size(self) -> None:
    assert_array_equal(self.g.fig.get_size_inches(), (10, 3))
    g = xplt.FacetGrid(self.darray, col='z', size=6)
    assert_array_equal(g.fig.get_size_inches(), (19, 6))
    g = self.darray.plot.imshow(col='z', size=6)
    assert_array_equal(g.fig.get_size_inches(), (19, 6))
    g = xplt.FacetGrid(self.darray, col='z', size=4, aspect=0.5)
    assert_array_equal(g.fig.get_size_inches(), (7, 4))
    g = xplt.FacetGrid(self.darray, col='z', figsize=(9, 4))
    assert_array_equal(g.fig.get_size_inches(), (9, 4))
    with pytest.raises(ValueError, match='cannot provide both'):
        g = xplt.plot(self.darray, row=2, col='z', figsize=(6, 4), size=6)
    with pytest.raises(ValueError, match="Can't use"):
        g = xplt.plot(self.darray, row=2, col='z', ax=plt.gca(), size=6)