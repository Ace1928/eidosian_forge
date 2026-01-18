import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def get_period_seconds(period):
    """
    return the number of seconds in the specified period

    >>> get_period_seconds('day')
    86400
    >>> get_period_seconds(86400)
    86400
    >>> get_period_seconds(datetime.timedelta(hours=24))
    86400
    >>> get_period_seconds('day + os.system("rm -Rf *")')
    Traceback (most recent call last):
    ...
    ValueError: period not in (second, minute, hour, day, month, year)
    """
    if isinstance(period, str):
        try:
            name = 'seconds_per_' + period.lower()
            result = globals()[name]
        except KeyError:
            msg = 'period not in (second, minute, hour, day, month, year)'
            raise ValueError(msg)
    elif isinstance(period, numbers.Number):
        result = period
    elif isinstance(period, datetime.timedelta):
        result = period.days * get_period_seconds('day') + period.seconds
    else:
        raise TypeError('period must be a string or integer')
    return result