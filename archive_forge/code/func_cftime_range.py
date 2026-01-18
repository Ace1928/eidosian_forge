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
def cftime_range(start=None, end=None, periods=None, freq=None, normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard'):
    """Return a fixed frequency CFTimeIndex.

    Parameters
    ----------
    start : str or cftime.datetime, optional
        Left bound for generating dates.
    end : str or cftime.datetime, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or None, default: "D"
        Frequency strings can have multiples, e.g. "5h" and negative values, e.g. "-1D".
    normalize : bool, default: False
        Normalize start/end dates to midnight before generating date range.
    name : str, default: None
        Name of the resulting index
    closed : {None, "left", "right"}, default: "NO_DEFAULT"
        Make the interval closed with respect to the given frequency to the
        "left", "right", or both sides (None).

        .. deprecated:: 2023.02.0
            Following pandas, the ``closed`` parameter is deprecated in favor
            of the ``inclusive`` parameter, and will be removed in a future
            version of xarray.

    inclusive : {None, "both", "neither", "left", "right"}, default None
        Include boundaries; whether to set each bound as closed or open.

        .. versionadded:: 2023.02.0

    calendar : str, default: "standard"
        Calendar type for the datetimes.

    Returns
    -------
    CFTimeIndex

    Notes
    -----
    This function is an analog of ``pandas.date_range`` for use in generating
    sequences of ``cftime.datetime`` objects.  It supports most of the
    features of ``pandas.date_range`` (e.g. specifying how the index is
    ``closed`` on either side, or whether or not to ``normalize`` the start and
    end bounds); however, there are some notable exceptions:

    - You cannot specify a ``tz`` (time zone) argument.
    - Start or end dates specified as partial-datetime strings must use the
      `ISO-8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_.
    - It supports many, but not all, frequencies supported by
      ``pandas.date_range``.  For example it does not currently support any of
      the business-related or semi-monthly frequencies.
    - Compound sub-monthly frequencies are not supported, e.g. '1H1min', as
      these can easily be written in terms of the finest common resolution,
      e.g. '61min'.

    Valid simple frequency strings for use with ``cftime``-calendars include
    any multiples of the following.

    +--------+--------------------------+
    | Alias  | Description              |
    +========+==========================+
    | YE     | Year-end frequency       |
    +--------+--------------------------+
    | YS     | Year-start frequency     |
    +--------+--------------------------+
    | QE     | Quarter-end frequency    |
    +--------+--------------------------+
    | QS     | Quarter-start frequency  |
    +--------+--------------------------+
    | ME     | Month-end frequency      |
    +--------+--------------------------+
    | MS     | Month-start frequency    |
    +--------+--------------------------+
    | D      | Day frequency            |
    +--------+--------------------------+
    | h      | Hour frequency           |
    +--------+--------------------------+
    | min    | Minute frequency         |
    +--------+--------------------------+
    | s      | Second frequency         |
    +--------+--------------------------+
    | ms     | Millisecond frequency    |
    +--------+--------------------------+
    | us     | Microsecond frequency    |
    +--------+--------------------------+

    Any multiples of the following anchored offsets are also supported.

    +------------+--------------------------------------------------------------------+
    | Alias      | Description                                                        |
    +============+====================================================================+
    | Y(E,S)-JAN | Annual frequency, anchored at the (end, beginning) of January      |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-FEB | Annual frequency, anchored at the (end, beginning) of February     |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-MAR | Annual frequency, anchored at the (end, beginning) of March        |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-APR | Annual frequency, anchored at the (end, beginning) of April        |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-MAY | Annual frequency, anchored at the (end, beginning) of May          |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-JUN | Annual frequency, anchored at the (end, beginning) of June         |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-JUL | Annual frequency, anchored at the (end, beginning) of July         |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-AUG | Annual frequency, anchored at the (end, beginning) of August       |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-SEP | Annual frequency, anchored at the (end, beginning) of September    |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-OCT | Annual frequency, anchored at the (end, beginning) of October      |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-NOV | Annual frequency, anchored at the (end, beginning) of November     |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-DEC | Annual frequency, anchored at the (end, beginning) of December     |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-JAN | Quarter frequency, anchored at the (end, beginning) of January     |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-FEB | Quarter frequency, anchored at the (end, beginning) of February    |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-MAR | Quarter frequency, anchored at the (end, beginning) of March       |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-APR | Quarter frequency, anchored at the (end, beginning) of April       |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-MAY | Quarter frequency, anchored at the (end, beginning) of May         |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-JUN | Quarter frequency, anchored at the (end, beginning) of June        |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-JUL | Quarter frequency, anchored at the (end, beginning) of July        |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-AUG | Quarter frequency, anchored at the (end, beginning) of August      |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-SEP | Quarter frequency, anchored at the (end, beginning) of September   |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-OCT | Quarter frequency, anchored at the (end, beginning) of October     |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-NOV | Quarter frequency, anchored at the (end, beginning) of November    |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-DEC | Quarter frequency, anchored at the (end, beginning) of December    |
    +------------+--------------------------------------------------------------------+

    Finally, the following calendar aliases are supported.

    +--------------------------------+---------------------------------------+
    | Alias                          | Date type                             |
    +================================+=======================================+
    | standard, gregorian            | ``cftime.DatetimeGregorian``          |
    +--------------------------------+---------------------------------------+
    | proleptic_gregorian            | ``cftime.DatetimeProlepticGregorian`` |
    +--------------------------------+---------------------------------------+
    | noleap, 365_day                | ``cftime.DatetimeNoLeap``             |
    +--------------------------------+---------------------------------------+
    | all_leap, 366_day              | ``cftime.DatetimeAllLeap``            |
    +--------------------------------+---------------------------------------+
    | 360_day                        | ``cftime.Datetime360Day``             |
    +--------------------------------+---------------------------------------+
    | julian                         | ``cftime.DatetimeJulian``             |
    +--------------------------------+---------------------------------------+

    Examples
    --------
    This function returns a ``CFTimeIndex``, populated with ``cftime.datetime``
    objects associated with the specified calendar type, e.g.

    >>> xr.cftime_range(start="2000", periods=6, freq="2MS", calendar="noleap")
    CFTimeIndex([2000-01-01 00:00:00, 2000-03-01 00:00:00, 2000-05-01 00:00:00,
                 2000-07-01 00:00:00, 2000-09-01 00:00:00, 2000-11-01 00:00:00],
                dtype='object', length=6, calendar='noleap', freq='2MS')

    As in the standard pandas function, three of the ``start``, ``end``,
    ``periods``, or ``freq`` arguments must be specified at a given time, with
    the other set to ``None``.  See the `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_
    for more examples of the behavior of ``date_range`` with each of the
    parameters.

    See Also
    --------
    pandas.date_range
    """
    if freq is None and any((arg is None for arg in [periods, start, end])):
        freq = 'D'
    if count_not_none(start, end, periods, freq) != 3:
        raise ValueError("Of the arguments 'start', 'end', 'periods', and 'freq', three must be specified at a time.")
    if start is not None:
        start = to_cftime_datetime(start, calendar)
        start = _maybe_normalize_date(start, normalize)
    if end is not None:
        end = to_cftime_datetime(end, calendar)
        end = _maybe_normalize_date(end, normalize)
    if freq is None:
        dates = _generate_linear_range(start, end, periods)
    else:
        offset = to_offset(freq)
        dates = np.array(list(_generate_range(start, end, periods, offset)))
    inclusive = _infer_inclusive(closed, inclusive)
    if inclusive == 'neither':
        left_closed = False
        right_closed = False
    elif inclusive == 'left':
        left_closed = True
        right_closed = False
    elif inclusive == 'right':
        left_closed = False
        right_closed = True
    elif inclusive == 'both':
        left_closed = True
        right_closed = True
    else:
        raise ValueError(f"Argument `inclusive` must be either 'both', 'neither', 'left', 'right', or None.  Got {inclusive}.")
    if not left_closed and len(dates) and (start is not None) and (dates[0] == start):
        dates = dates[1:]
    if not right_closed and len(dates) and (end is not None) and (dates[-1] == end):
        dates = dates[:-1]
    return CFTimeIndex(dates, name=name)