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
def test_coord_with_interval_x(self) -> None:
    """Test line plot with intervals explicitly on x axis."""
    bins = [-1, 0, 1, 2]
    self.darray.groupby_bins('dim_0', bins).mean(...).plot(x='dim_0_bins')