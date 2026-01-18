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
def add_trend(x, trend='c', prepend=False, has_constant='skip'):
    """
    Add a trend and/or constant to an array.

    Parameters
    ----------
    x : array_like
        Original array of data.
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add.

        * 'n' add no trend.
        * 'c' add constant only.
        * 't' add trend only.
        * 'ct' add constant and linear trend.
        * 'ctt' add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant column already
        exists in x. 'raise' will raise an error. 'add' will add a column of
        1s. 'skip' will return the data without change. 'skip' is the default.

    Returns
    -------
    array_like
        The original data with the additional trend columns.  If x is a
        pandas Series or DataFrame, then the trend column names are 'const',
        'trend' and 'trend_squared'.

    See Also
    --------
    statsmodels.tools.tools.add_constant
        Add a constant column to an array.

    Notes
    -----
    Returns columns as ['ctt','ct','c'] whenever applicable. There is currently
    no checking for an existing trend.
    """
    prepend = bool_like(prepend, 'prepend')
    trend = string_like(trend, 'trend', options=('n', 'c', 't', 'ct', 'ctt'))
    has_constant = string_like(has_constant, 'has_constant', options=('raise', 'add', 'skip'))
    columns = ['const', 'trend', 'trend_squared']
    if trend == 'n':
        return x.copy()
    elif trend == 'c':
        columns = columns[:1]
        trendorder = 0
    elif trend == 'ct' or trend == 't':
        columns = columns[:2]
        if trend == 't':
            columns = columns[1:2]
        trendorder = 1
    elif trend == 'ctt':
        trendorder = 2
    if _is_recarray(x):
        from statsmodels.tools.sm_exceptions import recarray_exception
        raise NotImplementedError(recarray_exception)
    is_pandas = _is_using_pandas(x, None)
    if is_pandas:
        if isinstance(x, pd.Series):
            x = pd.DataFrame(x)
        else:
            x = x.copy()
    else:
        x = np.asanyarray(x)
    nobs = len(x)
    trendarr = np.vander(np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1)
    trendarr = np.fliplr(trendarr)
    if trend == 't':
        trendarr = trendarr[:, 1]
    if 'c' in trend:
        if is_pandas:

            def safe_is_const(s):
                try:
                    return np.ptp(s) == 0.0 and np.any(s != 0.0)
                except:
                    return False
            col_const = x.apply(safe_is_const, 0)
        else:
            ptp0 = np.ptp(np.asanyarray(x), axis=0)
            col_is_const = ptp0 == 0
            nz_const = col_is_const & (x[0] != 0)
            col_const = nz_const
        if np.any(col_const):
            if has_constant == 'raise':
                if x.ndim == 1:
                    base_err = 'x is constant.'
                else:
                    columns = np.arange(x.shape[1])[col_const]
                    if isinstance(x, pd.DataFrame):
                        columns = x.columns
                    const_cols = ', '.join([str(c) for c in columns])
                    base_err = f'x contains one or more constant columns. Column(s) {const_cols} are constant.'
                msg = f"{base_err} Adding a constant with trend='{trend}' is not allowed."
                raise ValueError(msg)
            elif has_constant == 'skip':
                columns = columns[1:]
                trendarr = trendarr[:, 1:]
    order = 1 if prepend else -1
    if is_pandas:
        trendarr = pd.DataFrame(trendarr, index=x.index, columns=columns)
        x = [trendarr, x]
        x = pd.concat(x[::order], axis=1)
    else:
        x = [trendarr, x]
        x = np.column_stack(x[::order])
    return x