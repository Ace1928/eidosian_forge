import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def get_date_format_string(period):
    """
    For a given period (e.g. 'month', 'day', or some numeric interval
    such as 3600 (in secs)), return the format string that can be
    used with strftime to format that time to specify the times
    across that interval, but no more detailed.
    For example,

    >>> get_date_format_string('month')
    '%Y-%m'
    >>> get_date_format_string(3600)
    '%Y-%m-%d %H'
    >>> get_date_format_string('hour')
    '%Y-%m-%d %H'
    >>> get_date_format_string(None)
    Traceback (most recent call last):
        ...
    TypeError: period must be a string or integer
    >>> get_date_format_string('garbage')
    Traceback (most recent call last):
        ...
    ValueError: period not in (second, minute, hour, day, month, year)
    """
    if isinstance(period, str) and period.lower() == 'month':
        return '%Y-%m'
    file_period_secs = get_period_seconds(period)
    format_pieces = ('%Y', '-%m-%d', ' %H', '-%M', '-%S')
    seconds_per_second = 1
    intervals = (seconds_per_year, seconds_per_day, seconds_per_hour, seconds_per_minute, seconds_per_second)
    mods = list(map(lambda interval: file_period_secs % interval, intervals))
    format_pieces = format_pieces[:mods.index(0) + 1]
    return ''.join(format_pieces)