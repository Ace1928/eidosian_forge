from __future__ import annotations
import codecs
import functools
import inspect
import os
import re
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Set
from contextlib import contextmanager, nullcontext, suppress
from datetime import datetime, timedelta
from errno import ENOENT
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Integral, Number
from operator import add
from threading import Lock
from typing import Any, Callable, ClassVar, Literal, TypeVar, cast, overload
from weakref import WeakValueDictionary
import tlz as toolz
from dask import config
from dask.core import get_deps
from dask.typing import no_default
def format_time_ago(n: datetime) -> str:
    """Calculate a '3 hours ago' type string from a Python datetime.

    Examples
    --------
    >>> from datetime import datetime, timedelta

    >>> now = datetime.now()
    >>> format_time_ago(now)
    'Just now'

    >>> past = datetime.now() - timedelta(minutes=1)
    >>> format_time_ago(past)
    '1 minute ago'

    >>> past = datetime.now() - timedelta(minutes=2)
    >>> format_time_ago(past)
    '2 minutes ago'

    >>> past = datetime.now() - timedelta(hours=1)
    >>> format_time_ago(past)
    '1 hour ago'

    >>> past = datetime.now() - timedelta(hours=6)
    >>> format_time_ago(past)
    '6 hours ago'

    >>> past = datetime.now() - timedelta(days=1)
    >>> format_time_ago(past)
    '1 day ago'

    >>> past = datetime.now() - timedelta(days=5)
    >>> format_time_ago(past)
    '5 days ago'

    >>> past = datetime.now() - timedelta(days=8)
    >>> format_time_ago(past)
    '1 week ago'

    >>> past = datetime.now() - timedelta(days=16)
    >>> format_time_ago(past)
    '2 weeks ago'

    >>> past = datetime.now() - timedelta(days=190)
    >>> format_time_ago(past)
    '6 months ago'

    >>> past = datetime.now() - timedelta(days=800)
    >>> format_time_ago(past)
    '2 years ago'

    """
    units = {'years': lambda diff: diff.days / 365, 'months': lambda diff: diff.days / 30.436875, 'weeks': lambda diff: diff.days / 7, 'days': lambda diff: diff.days, 'hours': lambda diff: diff.seconds / 3600, 'minutes': lambda diff: diff.seconds % 3600 / 60}
    diff = datetime.now() - n
    for unit in units:
        dur = int(units[unit](diff))
        if dur > 0:
            if dur == 1:
                unit = unit[:-1]
            return f'{dur} {unit} ago'
    return 'Just now'