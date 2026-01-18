from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def month_anchor_check(dates):
    """Return the monthly offset string.

    Return "cs" if all dates are the first days of the month,
    "ce" if all dates are the last day of the month,
    None otherwise.

    Replicated pandas._libs.tslibs.resolution.month_position_check
    but without business offset handling.
    """
    calendar_end = True
    calendar_start = True
    for date in dates:
        if calendar_start:
            calendar_start &= date.day == 1
        if calendar_end:
            cal = date.day == date.daysinmonth
            calendar_end &= cal
        elif not calendar_start:
            break
    if calendar_end:
        return 'ce'
    elif calendar_start:
        return 'cs'
    else:
        return None