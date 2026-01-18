from __future__ import annotations
from collections.abc import (
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
import numpy as np
from pandas._config.config import (
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import (
from pandas.io.formats import printing
def get_format_timedelta64(values: TimedeltaArray, nat_rep: str | float='NaT', box: bool=False) -> Callable:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    even_days = values._is_dates_only
    if even_days:
        format = None
    else:
        format = 'long'

    def _formatter(x):
        if x is None or (is_scalar(x) and isna(x)):
            return nat_rep
        if not isinstance(x, Timedelta):
            x = Timedelta(x)
        result = x._repr_base(format=format)
        if box:
            result = f"'{result}'"
        return result
    return _formatter