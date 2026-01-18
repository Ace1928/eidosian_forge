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
def log_trans(base: Optional[float]=None, **kwargs: Any) -> trans:
    """
    Create a log transform class for *base*

    Parameters
    ----------
    base : float
        Base for the logarithm. If None, then
        the natural log is used.
    kwargs : dict
        Keyword arguments passed onto
        :func:`trans_new`. Should not include
        the `transform` or `inverse`.

    Returns
    -------
    out : type
        Log transform class
    """
    if base is None:
        name = 'log'
        base = np.exp(1)
        transform = np.log
    elif base == 10:
        name = 'log10'
        transform = np.log10
    elif base == 2:
        name = 'log2'
        transform = np.log2
    else:
        name = 'log{}'.format(base)

        def transform(x: FloatArrayLike) -> NDArrayFloat:
            return np.log(x) / np.log(base)

    def inverse(x):
        return np.power(base, x)
    if 'domain' not in kwargs:
        kwargs['domain'] = (sys.float_info.min, np.inf)
    if 'breaks' not in kwargs:
        kwargs['breaks'] = breaks_log(base=base)
    kwargs['base'] = base
    kwargs['_format'] = label_log(base)
    _trans = trans_new(name, transform, inverse, **kwargs)
    if 'minor_breaks' not in kwargs:
        n = int(base) - 2
        _trans.minor_breaks = minor_breaks_trans(_trans, n=n)
    return _trans