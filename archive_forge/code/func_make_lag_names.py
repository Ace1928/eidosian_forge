from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa
def make_lag_names(names, lag_order, trendorder=1, exog=None):
    """
    Produce list of lag-variable names. Constant / trends go at the beginning

    Examples
    --------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']
    """
    lag_names = []
    if isinstance(names, str):
        names = [names]
    for i in range(1, lag_order + 1):
        for name in names:
            if not isinstance(name, str):
                name = str(name)
            lag_names.append('L' + str(i) + '.' + name)
    if trendorder != 0:
        lag_names.insert(0, 'const')
    if trendorder > 1:
        lag_names.insert(1, 'trend')
    if trendorder > 2:
        lag_names.insert(2, 'trend**2')
    if exog is not None:
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        elif not hasattr(exog, 'ndim'):
            exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog[:, None]
        for i in range(exog.shape[1]):
            if isinstance(exog, pd.DataFrame):
                exog_name = str(exog.columns[i])
            else:
                exog_name = 'exog' + str(i)
            lag_names.insert(trendorder + i, exog_name)
    return lag_names