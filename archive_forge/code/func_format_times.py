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
def format_times(index, max_width, offset, separator=', ', first_row_offset=0, intermediate_row_end=',\n', last_row_end=''):
    """Format values of cftimeindex as pd.Index."""
    n_per_row = max(max_width // (CFTIME_REPR_LENGTH + len(separator)), 1)
    n_rows = math.ceil(len(index) / n_per_row)
    representation = ''
    for row in range(n_rows):
        indent = first_row_offset if row == 0 else offset
        row_end = last_row_end if row == n_rows - 1 else intermediate_row_end
        times_for_row = index[row * n_per_row:(row + 1) * n_per_row]
        representation += format_row(times_for_row, indent=indent, separator=separator, row_end=row_end)
    return representation