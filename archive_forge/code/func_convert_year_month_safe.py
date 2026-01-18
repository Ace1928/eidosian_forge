from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def convert_year_month_safe(year, month) -> Series:
    """
        Convert year and month to datetimes, using pandas vectorized versions
        when the date range falls within the range supported by pandas.
        Otherwise it falls back to a slower but more robust method
        using datetime.
        """
    if year.max() < MAX_YEAR and year.min() > MIN_YEAR:
        return to_datetime(100 * year + month, format='%Y%m')
    else:
        index = getattr(year, 'index', None)
        return Series([datetime(y, m, 1) for y, m in zip(year, month)], index=index)