from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_transform(data, fun, *args, **kwargs):
    """Perform a numerical transformation to data.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    fun : callable
        Numerical transformation function, `np.ufunc` or similar.
    args, kwargs : additional arguments, optional
        arguments for `fun`. `data` is always passed as the first argument

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Transformed output data
    """
    if is_sparse_dataframe(data) or is_SparseDataFrame(data):
        data = data.copy()
        for col in data.columns:
            data[col] = fun(data[col], *args, **kwargs)
    elif sparse.issparse(data):
        if isinstance(data, (sparse.lil_matrix, sparse.dok_matrix)):
            data = data.tocsr()
        else:
            data = data.copy()
        data.data = fun(data.data, *args, **kwargs)
    else:
        data = fun(data, *args, **kwargs)
    return data