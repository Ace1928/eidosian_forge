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
def test_datetime_hue(self) -> None:
    ds2 = self.ds.copy()
    ds2['hue'] = pd.date_range('2000-1-1', periods=4)
    ds2.plot.scatter(x='A', y='B', hue='hue')
    ds2['hue'] = pd.timedelta_range('-1D', periods=4, freq='D')
    ds2.plot.scatter(x='A', y='B', hue='hue')