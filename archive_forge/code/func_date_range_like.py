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
def date_range_like(source, calendar, use_cftime=None):
    """Generate a datetime array with the same frequency, start and end as
    another one, but in a different calendar.

    Parameters
    ----------
    source : DataArray, CFTimeIndex, or pd.DatetimeIndex
        1D datetime array
    calendar : str
        New calendar name.
    use_cftime : bool, optional
        If True, the output uses :py:class:`cftime.datetime` objects.
        If None (default), :py:class:`numpy.datetime64` values are used if possible.
        If False, :py:class:`numpy.datetime64` values are used or an error is raised.

    Returns
    -------
    DataArray
        1D datetime coordinate with the same start, end and frequency as the
        source, but in the new calendar. The start date is assumed to exist in
        the target calendar. If the end date doesn't exist, the code tries 1
        and 2 calendar days before. There is a special case when the source time
        series is daily or coarser and the end of the input range is on the
        last day of the month. Then the output range will also end on the last
        day of the month in the new calendar.
    """
    from xarray.coding.frequencies import infer_freq
    from xarray.core.dataarray import DataArray
    if not isinstance(source, (pd.DatetimeIndex, CFTimeIndex)) and (isinstance(source, DataArray) and source.ndim != 1 or not _contains_datetime_like_objects(source.variable)):
        raise ValueError("'source' must be a 1D array of datetime objects for inferring its range.")
    freq = infer_freq(source)
    if freq is None:
        raise ValueError('`date_range_like` was unable to generate a range as the source frequency was not inferable.')
    freq = _legacy_to_new_freq(freq)
    use_cftime = _should_cftime_be_used(source, calendar, use_cftime)
    source_start = source.values.min()
    source_end = source.values.max()
    freq_as_offset = to_offset(freq)
    if freq_as_offset.n < 0:
        source_start, source_end = (source_end, source_start)
    if is_np_datetime_like(source.dtype):
        source_calendar = 'standard'
        source_start = nanosecond_precision_timestamp(source_start)
        source_end = nanosecond_precision_timestamp(source_end)
    elif isinstance(source, CFTimeIndex):
        source_calendar = source.calendar
    else:
        source_calendar = source.dt.calendar
    if calendar == source_calendar and is_np_datetime_like(source.dtype) ^ use_cftime:
        return source
    date_type = get_date_type(calendar, use_cftime)
    start = convert_time_or_go_back(source_start, date_type)
    end = convert_time_or_go_back(source_end, date_type)
    if source_end.day == source_end.daysinmonth and isinstance(freq_as_offset, (YearEnd, QuarterEnd, MonthEnd, Day)):
        end = end.replace(day=end.daysinmonth)
    return date_range(start=start.isoformat(), end=end.isoformat(), freq=freq, calendar=calendar)