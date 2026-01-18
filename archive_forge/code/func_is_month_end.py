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
def is_month_end(self) -> T_DataArray:
    """Indicate whether the date is the last day of the month"""
    return self._date_field('is_month_end', bool)