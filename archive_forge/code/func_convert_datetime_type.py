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
def convert_datetime_type(obj: Any | pd.Timestamp | pd.Timedelta | dt.datetime | dt.date | dt.time | np.datetime64) -> float:
    """ Convert any recognized date, time, or datetime value to floating point
    milliseconds since epoch.

    Args:
        obj (object) : the object to convert

    Returns:
        float : milliseconds

    """
    import pandas as pd
    if obj is pd.NaT:
        return np.nan
    if isinstance(obj, pd.Period):
        return obj.to_timestamp().value / 10 ** 6.0
    if isinstance(obj, pd.Timestamp):
        return obj.value / 10 ** 6.0
    elif isinstance(obj, pd.Timedelta):
        return obj.value / 10 ** 6.0
    elif isinstance(obj, dt.datetime):
        diff = obj.replace(tzinfo=dt.timezone.utc) - DT_EPOCH
        return diff.total_seconds() * 1000
    elif isinstance(obj, dt.date):
        return convert_date_to_datetime(obj)
    elif isinstance(obj, np.datetime64):
        epoch_delta = obj - NP_EPOCH
        return float(epoch_delta / NP_MS_DELTA)
    elif isinstance(obj, dt.time):
        return (obj.hour * 3600 + obj.minute * 60 + obj.second) * 1000 + obj.microsecond / 1000.0
    raise ValueError(f'unknown datetime object: {obj!r}')