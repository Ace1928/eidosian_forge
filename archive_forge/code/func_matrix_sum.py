from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_sum(data, axis=None, ignore_nan=False):
    """Get the column-wise, row-wise, or total sum of values in a matrix.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    axis : int or None, optional (default: None)
        Axis across which to sum. axis=0 gives column sums,
        axis=1 gives row sums. None gives the total sum.
    ignore_nan : bool, optional (default: False)
        If True, uses `np.nansum` instead of `np.sum`

    Returns
    -------
    sums : array-like or float
        Sums along desired axis.
    """
    sum_fn = _nansum if ignore_nan else np.sum
    if axis not in [0, 1, None]:
        raise ValueError('Expected axis in [0, 1, None]. Got {}'.format(axis))
    if isinstance(data, pd.DataFrame):
        if is_SparseDataFrame(data):
            if axis is None:
                sums = sum_fn(data.to_coo())
            else:
                index = data.index if axis == 1 else data.columns
                sums = pd.Series(np.array(sum_fn(data.to_coo(), axis)).flatten(), index=index)
        elif is_sparse_dataframe(data):
            if axis is None:
                sums = sum_fn(data.sparse.to_coo())
            else:
                index = data.index if axis == 1 else data.columns
                sums = pd.Series(np.array(sum_fn(data.sparse.to_coo(), axis)).flatten(), index=index)
        elif axis is None:
            sums = sum_fn(data.to_numpy())
        else:
            sums = sum_fn(data, axis)
    else:
        sums = sum_fn(data, axis=axis)
        if isinstance(sums, np.matrix):
            sums = np.array(sums).flatten()
    return sums