from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
def add_lag(x, col=None, lags=1, drop=False, insert=True):
    """
    Returns an array with lags included given an array.

    Parameters
    ----------
    x : array_like
        An array or NumPy ndarray subclass. Can be either a 1d or 2d array with
        observations in columns.
    col : int or None
        `col` can be an int of the zero-based column index. If it's a
        1d array `col` can be None.
    lags : int
        The number of lags desired.
    drop : bool
        Whether to keep the contemporaneous variable for the data.
    insert : bool or int
        If True, inserts the lagged values after `col`. If False, appends
        the data. If int inserts the lags at int.

    Returns
    -------
    array : ndarray
        Array with lags

    Examples
    --------

    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load()
    >>> data = data.data[['year','quarter','realgdp','cpi']]
    >>> data = sm.tsa.add_lag(data, 'realgdp', lags=2)

    Notes
    -----
    Trims the array both forward and backward, so that the array returned
    so that the length of the returned array is len(`X`) - lags. The lags are
    returned in increasing order, ie., t-1,t-2,...,t-lags
    """
    lags = int_like(lags, 'lags')
    drop = bool_like(drop, 'drop')
    x = array_like(x, 'x', ndim=2)
    if col is None:
        col = 0
    if col < 0:
        col = x.shape[1] + col
    if x.ndim == 1:
        x = x[:, None]
    contemp = x[:, col]
    if insert is True:
        ins_idx = col + 1
    elif insert is False:
        ins_idx = x.shape[1]
    else:
        if insert < 0:
            insert = x.shape[1] + insert + 1
        if insert > x.shape[1]:
            insert = x.shape[1]
            warnings.warn('insert > number of variables, inserting at the last position', ValueWarning)
        ins_idx = insert
    ndlags = lagmat(contemp, lags, trim='Both')
    first_cols = lrange(ins_idx)
    last_cols = lrange(ins_idx, x.shape[1])
    if drop:
        if col in first_cols:
            first_cols.pop(first_cols.index(col))
        else:
            last_cols.pop(last_cols.index(col))
    return np.column_stack((x[lags:, first_cols], ndlags, x[lags:, last_cols]))