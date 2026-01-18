from __future__ import annotations
import collections
import datetime as dt
import functools
import inspect
import json
import re
import typing
import warnings
from collections.abc import Mapping
from email.utils import format_datetime, parsedate_to_datetime
from pprint import pprint as py_pprint
from marshmallow.base import FieldABC
from marshmallow.exceptions import FieldInstanceResolutionError
from marshmallow.warnings import RemovedInMarshmallow4Warning
def from_iso_time(value):
    """Parse a string and return a datetime.time.

    This function doesn't support time zone offsets.
    """
    match = _iso8601_time_re.match(value)
    if not match:
        raise ValueError('Not a valid ISO8601-formatted time string')
    kw = match.groupdict()
    kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
    kw = {k: int(v) for k, v in kw.items() if v is not None}
    return dt.time(**kw)