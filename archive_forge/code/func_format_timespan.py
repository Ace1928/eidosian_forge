import collections
import datetime
import decimal
import numbers
import os
import os.path
import re
import time
from humanfriendly.compat import is_string, monotonic
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format, pluralize, tokenize
def format_timespan(num_seconds, detailed=False, max_units=3):
    """
    Format a timespan in seconds as a human readable string.

    :param num_seconds: Any value accepted by :func:`coerce_seconds()`.
    :param detailed: If :data:`True` milliseconds are represented separately
                     instead of being represented as fractional seconds
                     (defaults to :data:`False`).
    :param max_units: The maximum number of units to show in the formatted time
                      span (an integer, defaults to three).
    :returns: The formatted timespan as a string.
    :raise: See :func:`coerce_seconds()`.

    Some examples:

    >>> from humanfriendly import format_timespan
    >>> format_timespan(0)
    '0 seconds'
    >>> format_timespan(1)
    '1 second'
    >>> import math
    >>> format_timespan(math.pi)
    '3.14 seconds'
    >>> hour = 60 * 60
    >>> day = hour * 24
    >>> week = day * 7
    >>> format_timespan(week * 52 + day * 2 + hour * 3)
    '1 year, 2 days and 3 hours'
    """
    num_seconds = coerce_seconds(num_seconds)
    if num_seconds < 60 and (not detailed):
        return pluralize(round_number(num_seconds), 'second')
    else:
        result = []
        num_seconds = decimal.Decimal(str(num_seconds))
        relevant_units = list(reversed(time_units[0 if detailed else 3:]))
        for unit in relevant_units:
            divider = decimal.Decimal(str(unit['divider']))
            count = num_seconds / divider
            num_seconds %= divider
            if unit != relevant_units[-1]:
                count = int(count)
            else:
                count = round_number(count)
            if count not in (0, '0'):
                result.append(pluralize(count, unit['singular'], unit['plural']))
        if len(result) == 1:
            return result[0]
        else:
            if not detailed:
                result = result[:max_units]
            return concatenate(result)