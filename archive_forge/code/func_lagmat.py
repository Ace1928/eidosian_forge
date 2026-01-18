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
def lagmat(x, maxlag: int, trim: Literal['forward', 'backward', 'both', 'none']='forward', original: Literal['ex', 'sep', 'in']='ex', use_pandas: bool=False) -> NDArray | DataFrame | tuple[NDArray, NDArray] | tuple[DataFrame, DataFrame]:
    """
    Create 2d array of lags.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none', None}
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none', None : no trimming of observations.
    original : {'ex','sep','in'}
        How the original is treated.

        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray
        The array with lagged observations.
    y : ndarray, optional
        Only returned if original == 'sep'.

    Notes
    -----
    When using a pandas DataFrame or Series with use_pandas=True, trim can only
    be 'forward' or 'both' since it is not possible to consistently extend
    index values.

    Examples
    --------
    >>> from statsmodels.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])
    """
    maxlag = int_like(maxlag, 'maxlag')
    use_pandas = bool_like(use_pandas, 'use_pandas')
    trim = string_like(trim, 'trim', optional=True, options=('forward', 'backward', 'both', 'none'))
    original = string_like(original, 'original', options=('ex', 'sep', 'in'))
    orig = x
    x = array_like(x, 'x', ndim=2, dtype=None)
    is_pandas = _is_using_pandas(orig, None) and use_pandas
    trim = 'none' if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ('none', 'backward'):
        raise ValueError("trim cannot be 'none' or 'backward' when used on Series or DataFrames")
    dropidx = 0
    nobs, nvar = x.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError('maxlag should be < nobs')
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k, nvar * (maxlag - k):nvar * (maxlag - k + 1)] = x
    if trim in ('none', 'forward'):
        startobs = 0
    elif trim in ('backward', 'both'):
        startobs = maxlag
    else:
        raise ValueError('trim option not valid')
    if trim in ('none', 'backward'):
        stopobs = len(lm)
    else:
        stopobs = nobs
    if is_pandas:
        x = orig
        if isinstance(x, DataFrame):
            x_columns = [str(c) for c in x.columns]
            if len(set(x_columns)) != x.shape[1]:
                raise ValueError('Columns names must be distinct after conversion to string (if not already strings).')
        else:
            x_columns = [str(x.name)]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + '.L.' + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ('sep', 'ex'):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, axis=1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == 'sep':
            leads = lm[startobs:stopobs, :dropidx]
    if original == 'sep':
        return (lags, leads)
    else:
        return lags