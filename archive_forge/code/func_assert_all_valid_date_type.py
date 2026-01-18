from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def assert_all_valid_date_type(data):
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    if len(data) > 0:
        sample = data[0]
        date_type = type(sample)
        if not isinstance(sample, cftime.datetime):
            raise TypeError(f'CFTimeIndex requires cftime.datetime objects. Got object of {date_type}.')
        if not all((isinstance(value, date_type) for value in data)):
            raise TypeError(f'CFTimeIndex requires using datetime objects of all the same type.  Got\n{data}.')