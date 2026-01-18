from __future__ import annotations
import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import parse_http_list as _parse_list_header
from ._internal import _dt_as_utc
from ._internal import _plain_int
from . import datastructures as ds
from .sansio import http as _sansio_http
def parse_content_range_header(value: str | None, on_update: t.Callable[[ds.ContentRange], None] | None=None) -> ds.ContentRange | None:
    """Parses a range header into a
    :class:`~werkzeug.datastructures.ContentRange` object or `None` if
    parsing is not possible.

    .. versionadded:: 0.7

    :param value: a content range header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.ContentRange`
                      object is changed.
    """
    if value is None:
        return None
    try:
        units, rangedef = (value or '').strip().split(None, 1)
    except ValueError:
        return None
    if '/' not in rangedef:
        return None
    rng, length_str = rangedef.split('/', 1)
    if length_str == '*':
        length = None
    else:
        try:
            length = _plain_int(length_str)
        except ValueError:
            return None
    if rng == '*':
        if not is_byte_range_valid(None, None, length):
            return None
        return ds.ContentRange(units, None, None, length, on_update=on_update)
    elif '-' not in rng:
        return None
    start_str, stop_str = rng.split('-', 1)
    try:
        start = _plain_int(start_str)
        stop = _plain_int(stop_str) + 1
    except ValueError:
        return None
    if is_byte_range_valid(start, stop, length):
        return ds.ContentRange(units, start, stop, length, on_update=on_update)
    return None