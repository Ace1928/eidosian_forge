import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def drop_missing(Y, X=None, axis=1):
    """
    Returns views on the arrays Y and X where missing observations are dropped.

    Y : array_like
    X : array_like, optional
    axis : int
        Axis along which to look for missing observations.  Default is 1, ie.,
        observations in rows.

    Returns
    -------
    Y : ndarray
        All Y where the
    X : ndarray

    Notes
    -----
    If either Y or X is 1d, it is reshaped to be 2d.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X is not None:
        X = np.array(X)
        if X.ndim == 1:
            X = X[:, None]
        keepidx = np.logical_and(~np.isnan(Y).any(axis), ~np.isnan(X).any(axis))
        return (Y[keepidx], X[keepidx])
    else:
        keepidx = ~np.isnan(Y).any(axis)
        return Y[keepidx]