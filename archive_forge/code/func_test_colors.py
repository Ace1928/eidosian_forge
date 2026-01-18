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
def test_colors(self) -> None:
    artist = self.plotmethod(colors='k')
    assert artist.cmap.colors[0] == 'k'
    artist = self.plotmethod(colors=['k', 'b'])
    assert self._color_as_tuple(artist.cmap.colors[1]) == (0.0, 0.0, 1.0)
    artist = self.darray.plot.contour(levels=[-0.5, 0.0, 0.5, 1.0], colors=['k', 'r', 'w', 'b'])
    assert self._color_as_tuple(artist.cmap.colors[1]) == (1.0, 0.0, 0.0)
    assert self._color_as_tuple(artist.cmap.colors[2]) == (1.0, 1.0, 1.0)
    assert self._color_as_tuple(artist.cmap._rgba_over) == (0.0, 0.0, 1.0)