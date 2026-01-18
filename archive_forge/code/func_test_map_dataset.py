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
def test_map_dataset(self) -> None:
    g = xplt.FacetGrid(self.darray.to_dataset(name='foo'), col='z')
    g.map(plt.contourf, 'x', 'y', 'foo')
    alltxt = text_in_fig()
    for label in ['x', 'y']:
        assert label in alltxt
    assert 'None' not in alltxt
    assert 'foo' not in alltxt
    assert 0 == len(find_possible_colorbars())
    g.add_colorbar(label='colors!')
    assert 'colors!' in text_in_fig()
    assert 1 == len(find_possible_colorbars())