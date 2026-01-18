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
def test_discrete_colormap_list_of_levels(self) -> None:
    for extend, levels in [('max', [-1, 2, 4, 8, 10]), ('both', [2, 5, 10, 11]), ('neither', [0, 5, 10, 15]), ('min', [2, 5, 10, 15])]:
        for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
            primitive = getattr(self.darray.plot, kind)(levels=levels)
            assert_array_equal(levels, primitive.norm.boundaries)
            assert max(levels) == primitive.norm.vmax
            assert min(levels) == primitive.norm.vmin
            if kind != 'contour':
                assert extend == primitive.cmap.colorbar_extend
            else:
                assert 'max' == primitive.cmap.colorbar_extend
            assert len(levels) - 1 == len(primitive.cmap.colors)