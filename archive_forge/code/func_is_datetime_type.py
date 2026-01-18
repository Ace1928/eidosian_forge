from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def is_datetime_type(obj: Any) -> TypeGuard[dt.time | dt.datetime | np.datetime64]:
    """ Whether an object is any date, time, or datetime type recognized by
    Bokeh.

    Args:
        obj (object) : the object to test

    Returns:
        bool : True if ``obj`` is a datetime type

    """
    _dt_tuple = tuple(_compute_datetime_types())
    return isinstance(obj, _dt_tuple)