from . import measure
from . import utils
from scipy import sparse
from sklearn.preprocessing import normalize
import numbers
import numpy as np
import pandas as pd
import warnings
def library_size_normalize(data, rescale=10000, return_library_size=False):
    """Perform L1 normalization on input data.

    Performs L1 normalization on input data such that the sum of expression
    values for each cell sums to 1
    then returns normalized matrix to the metric space using median UMI count
    per cell effectively scaling all cells as if they were sampled evenly.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    rescale : {'mean', 'median'}, float or `None`, optional (default: 10000)
        Rescaling strategy. If 'mean' or 'median', normalized cells are scaled
        back up to the mean or median expression value. If a float,
        normalized cells are scaled up to the given value. If `None`, no
        rescaling is done and all cells will have normalized library size of 1.
    return_library_size : bool, optional (default: False)
        If True, also return the library size pre-normalization

    Returns
    -------
    data_norm : array-like, shape=[n_samples, n_features]
        Library size normalized output data
    filtered_library_size : list-like, shape=[m_samples]
        Library size of cells pre-normalization,
        returned only if return_library_size is True
    """
    columns, index = (None, None)
    if isinstance(data, pd.DataFrame):
        columns, index = (data.columns, data.index)
        if utils.is_sparse_dataframe(data):
            data = data.sparse.to_coo()
        elif utils.is_SparseDataFrame(data):
            data = data.to_coo()
        else:
            data = data.to_numpy()
    calc_libsize = sparse.issparse(data) and (return_library_size or data.nnz > 2 ** 31)
    rescale, libsize = _get_scaled_libsize(data, rescale, calc_libsize)
    if libsize is not None:
        divisor = utils.toarray(libsize)
        data_norm = utils.matrix_vector_elementwise_multiply(data, 1 / np.where(divisor == 0, 1, divisor), axis=0)
    elif return_library_size:
        data_norm, libsize = normalize(data, norm='l1', axis=1, return_norm=True)
    else:
        data_norm = normalize(data, norm='l1', axis=1)
    data_norm = data_norm * rescale
    if columns is not None:
        if sparse.issparse(data_norm):
            data_norm = utils.SparseDataFrame(data_norm, default_fill_value=0.0)
        else:
            data_norm = pd.DataFrame(data_norm)
        data_norm.columns = columns
        data_norm.index = index
        libsize = pd.Series(libsize, index=index, name='library_size', dtype='float64')
    if return_library_size:
        return (data_norm, libsize)
    else:
        return data_norm