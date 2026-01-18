from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Generic
import numpy as np
import pandas as pd
from xarray.coding.times import infer_calendar_name
from xarray.core import duck_array_ops
from xarray.core.common import (
from xarray.core.types import T_DataArray
from xarray.core.variable import IndexVariable
from xarray.namedarray.utils import is_duck_dask_array
@property
def days_in_month(self) -> T_DataArray:
    """The number of days in the month"""
    return self._date_field('days_in_month', np.int64)