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
def test_2d_coord_with_interval(self) -> None:
    for dim in self.darray.dims:
        gp = self.darray.groupby_bins(dim, range(15), restore_coord_dims=True).mean([dim])
        for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
            getattr(gp.plot, kind)()