from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_vector_elementwise_multiply(data, multiplier, axis=None):
    """Elementwise multiply a matrix by a vector.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    multiplier : array-like, shape=[n_samples, 1] or [1, n_features]
        Vector by which to multiply `data`
    axis : int or None, optional (default: None)
        Axis across which to sum. axis=0 multiplies each column,
        axis=1 multiplies each row. None guesses based on dimensions

    Returns
    -------
    product : array-like
        Multiplied matrix
    """
    if axis not in [0, 1, None]:
        raise ValueError('Expected axis in [0, 1, None]. Got {}'.format(axis))
    if axis is None:
        if data.shape[0] == data.shape[1]:
            raise RuntimeError('`data` is square, cannot guess axis from input. Please provide `axis=0` to multiply along rows or `axis=1` to multiply along columns.')
        elif np.prod(multiplier.shape) == data.shape[0]:
            axis = 0
        elif np.prod(multiplier.shape) == data.shape[1]:
            axis = 1
        else:
            raise ValueError('Expected `multiplier` to be a vector of length `data.shape[0]` ({}) or `data.shape[1]` ({}). Got {}'.format(data.shape[0], data.shape[1], multiplier.shape))
    multiplier = toarray(multiplier)
    if axis == 0:
        if not np.prod(multiplier.shape) == data.shape[0]:
            raise ValueError('Expected `multiplier` to be a vector of length `data.shape[0]` ({}). Got {}'.format(data.shape[0], multiplier.shape))
        multiplier = multiplier.reshape(-1, 1)
    else:
        if not np.prod(multiplier.shape) == data.shape[1]:
            raise ValueError('Expected `multiplier` to be a vector of length `data.shape[1]` ({}). Got {}'.format(data.shape[1], multiplier.shape))
        multiplier = multiplier.reshape(1, -1)
    if is_SparseDataFrame(data) or is_sparse_dataframe(data):
        data = data.copy()
        multiplier = multiplier.flatten()
        if axis == 0:
            for col in data.columns:
                try:
                    mult_indices = data[col].values.sp_index.indices
                except AttributeError:
                    mult_indices = data[col].values.sp_index.to_int_index().indices
                new_data = data[col].values.sp_values * multiplier[mult_indices]
                data[col].values.sp_values.put(np.arange(data[col].sparse.npoints), new_data)
        else:
            for col, mult in zip(data.columns, multiplier):
                data[col] = data[col] * mult
    elif isinstance(data, pd.DataFrame):
        data = data.mul(multiplier.flatten(), axis=axis)
    elif sparse.issparse(data):
        if isinstance(data, (sparse.lil_matrix, sparse.dok_matrix, sparse.coo_matrix, sparse.bsr_matrix, sparse.dia_matrix)):
            data = data.tocsr()
        data = data.multiply(multiplier)
    else:
        data = data * multiplier
    return data