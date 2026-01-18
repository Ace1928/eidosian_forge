from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_std(data, axis=None):
    """Get the column-wise, row-wise, or total standard deviation of a matrix.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    axis : int or None, optional (default: None)
        Axis across which to calculate standard deviation.
        axis=0 gives column standard deviation,
        axis=1 gives row standard deviation.
        None gives the total standard deviation.

    Returns
    -------
    std : array-like or float
        Standard deviation along desired axis.
    """
    if axis not in [0, 1, None]:
        raise ValueError('Expected axis in [0, 1, None]. Got {}'.format(axis))
    index = None
    if isinstance(data, pd.DataFrame) and axis is not None:
        if axis == 1:
            index = data.index
        elif axis == 0:
            index = data.columns
    data = to_array_or_spmatrix(data)
    if sparse.issparse(data):
        if axis is None:
            if isinstance(data, (sparse.lil_matrix, sparse.dok_matrix)):
                data = data.tocoo()
            data_sq = data.copy()
            data_sq.data = data_sq.data ** 2
            variance = data_sq.mean() - data.mean() ** 2
            std = np.sqrt(variance)
        else:
            if axis == 0:
                data = data.tocsc()
                next_fn = data.getcol
                N = data.shape[1]
            elif axis == 1:
                data = data.tocsr()
                next_fn = data.getrow
                N = data.shape[0]
            std = []
            for i in range(N):
                col = next_fn(i)
                col_sq = col.copy()
                col_sq.data = col_sq.data ** 2
                variance = col_sq.mean() - col.mean() ** 2
                std.append(np.sqrt(variance))
            std = np.array(std)
    else:
        std = np.std(data, axis=axis)
    if index is not None:
        std = pd.Series(std, index=index, name='std')
    return std