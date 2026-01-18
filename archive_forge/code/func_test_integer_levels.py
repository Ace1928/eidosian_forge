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
def test_integer_levels(self) -> None:
    data = self.data + 1
    for level in np.arange(2, 10, dtype=int):
        cmap_params = _determine_cmap_params(data, levels=level)
        assert cmap_params['vmin'] is None
        assert cmap_params['vmax'] is None
        assert cmap_params['norm'].vmin == cmap_params['levels'][0]
        assert cmap_params['norm'].vmax == cmap_params['levels'][-1]
        assert cmap_params['extend'] == 'neither'
    cmap_params = _determine_cmap_params(data, levels=5, vmin=0, vmax=5, cmap='Blues')
    assert cmap_params['vmin'] is None
    assert cmap_params['vmax'] is None
    assert cmap_params['norm'].vmin == 0
    assert cmap_params['norm'].vmax == 5
    assert cmap_params['norm'].vmin == cmap_params['levels'][0]
    assert cmap_params['norm'].vmax == cmap_params['levels'][-1]
    assert cmap_params['cmap'].name == 'Blues'
    assert cmap_params['extend'] == 'neither'
    assert cmap_params['cmap'].N == 4
    assert cmap_params['norm'].N == 5
    cmap_params = _determine_cmap_params(data, levels=5, vmin=0.5, vmax=1.5)
    assert cmap_params['cmap'].name == 'viridis'
    assert cmap_params['extend'] == 'max'
    cmap_params = _determine_cmap_params(data, levels=5, vmin=1.5)
    assert cmap_params['cmap'].name == 'viridis'
    assert cmap_params['extend'] == 'min'
    cmap_params = _determine_cmap_params(data, levels=5, vmin=1.3, vmax=1.5)
    assert cmap_params['cmap'].name == 'viridis'
    assert cmap_params['extend'] == 'both'