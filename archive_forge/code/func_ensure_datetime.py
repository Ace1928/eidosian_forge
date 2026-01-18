import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def ensure_datetime(ob: AnyDatetime) -> datetime.datetime:
    """
    Given a datetime or date or time object from the ``datetime``
    module, always return a datetime using default values.
    """
    if isinstance(ob, datetime.datetime):
        return ob
    date = cast(datetime.date, ob)
    time = cast(datetime.time, ob)
    if isinstance(ob, datetime.date):
        time = datetime.time()
    if isinstance(ob, datetime.time):
        date = datetime.date(1900, 1, 1)
    return datetime.datetime.combine(date, time)