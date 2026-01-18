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
def parse_timespan(timespan):
    """
    Parse a "human friendly" timespan into the number of seconds.

    :param value: A string like ``5h`` (5 hours), ``10m`` (10 minutes) or
                  ``42s`` (42 seconds).
    :returns: The number of seconds as a floating point number.
    :raises: :exc:`InvalidTimespan` when the input can't be parsed.

    Note that the :func:`parse_timespan()` function is not meant to be the
    "mirror image" of the :func:`format_timespan()` function. Instead it's
    meant to allow humans to easily and succinctly specify a timespan with a
    minimal amount of typing. It's very useful to accept easy to write time
    spans as e.g. command line arguments to programs.

    The time units (and abbreviations) supported by this function are:

    - ms, millisecond, milliseconds
    - s, sec, secs, second, seconds
    - m, min, mins, minute, minutes
    - h, hour, hours
    - d, day, days
    - w, week, weeks
    - y, year, years

    Some examples:

    >>> from humanfriendly import parse_timespan
    >>> parse_timespan('42')
    42.0
    >>> parse_timespan('42s')
    42.0
    >>> parse_timespan('1m')
    60.0
    >>> parse_timespan('1h')
    3600.0
    >>> parse_timespan('1d')
    86400.0
    """
    tokens = tokenize(timespan)
    if tokens and isinstance(tokens[0], numbers.Number):
        if len(tokens) == 1:
            return float(tokens[0])
        if len(tokens) == 2 and is_string(tokens[1]):
            normalized_unit = tokens[1].lower()
            for unit in time_units:
                if normalized_unit == unit['singular'] or normalized_unit == unit['plural'] or normalized_unit in unit['abbreviations']:
                    return float(tokens[0]) * unit['divider']
    msg = 'Failed to parse timespan! (input %r was tokenized as %r)'
    raise InvalidTimespan(format(msg, timespan, tokens))