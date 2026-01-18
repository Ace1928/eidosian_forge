from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import (
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import (
from xarray.core.utils import emit_user_level_warning
def roll_qtrday(other, n, month, day_option, modby=3):
    """Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : cftime.datetime
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_option : 'start', 'end'
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    _get_day_of_month : Find the day in a month provided an offset.
    """
    months_since = other.month % modby - month % modby
    if n > 0:
        if months_since < 0 or (months_since == 0 and other.day < _get_day_of_month(other, day_option)):
            n -= 1
    elif months_since > 0 or (months_since == 0 and other.day > _get_day_of_month(other, day_option)):
        n += 1
    return n