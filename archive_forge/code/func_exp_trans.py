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
def exp_trans(base: Optional[float]=None, **kwargs: Any):
    """
    Create a exponential transform class for *base*

    This is inverse of the log transform.

    Parameters
    ----------
    base : float
        Base of the logarithm
    kwargs : dict
        Keyword arguments passed onto
        :func:`trans_new`. Should not include
        the `transform` or `inverse`.

    Returns
    -------
    out : type
        Exponential transform class
    """
    if base is None:
        name = 'power_e'
        base = np.exp(1)
    else:
        name = 'power_{}'.format(base)

    def transform(x):
        return np.power(base, x)

    def inverse(x):
        return np.log(x) / np.log(base)
    kwargs['base'] = base
    return trans_new(name, transform, inverse, **kwargs)