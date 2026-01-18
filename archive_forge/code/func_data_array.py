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
@pytest.fixture(params=[1, 2, 3])
def data_array(self, request):
    """
        Return a simple DataArray
        """
    dims = request.param
    if dims == 1:
        return DataArray(easy_array((10,)))
    if dims == 2:
        return DataArray(easy_array((10, 3)))
    if dims == 3:
        return DataArray(easy_array((10, 3, 2)))