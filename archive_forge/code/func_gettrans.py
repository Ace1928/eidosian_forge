from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
def gettrans(t: str | Callable[[], Type[trans]] | Type[trans] | trans):
    """
    Return a trans object

    Parameters
    ----------
    t : str | callable | type | trans
        name of transformation function

    Returns
    -------
    out : trans
    """
    obj = t
    if isinstance(obj, str):
        name = '{}_trans'.format(obj)
        obj = globals()[name]()
    if callable(obj):
        obj = obj()
    if isinstance(obj, type):
        obj = obj()
    if not isinstance(obj, trans):
        raise ValueError('Could not get transform object.')
    return obj