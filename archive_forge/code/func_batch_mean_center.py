from . import measure
from . import utils
from scipy import sparse
from sklearn.preprocessing import normalize
import numbers
import numpy as np
import pandas as pd
import warnings
def batch_mean_center(data, sample_idx=None):
    """Perform batch mean-centering on the data.

    The features of the data are all centered such that
    the column means are zero. Each batch is centered separately.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    sample_idx : list-like, optional
        Batch indices. If `None`, data is assumed to be a single batch

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Batch mean-centered output data.
    """
    if sparse.issparse(data) or utils.is_SparseDataFrame(data) or utils.is_sparse_dataframe(data):
        raise ValueError('Cannot mean center sparse data. Convert to dense matrix first.')
    if sample_idx is None:
        sample_idx = np.ones(len(data))
    else:
        sample_idx = utils.toarray(sample_idx).flatten()
    for sample in np.unique(sample_idx):
        idx = sample_idx == sample
        if isinstance(data, pd.DataFrame):
            feature_means = data.iloc[idx].mean(axis=0)
            data.iloc[idx] = data.iloc[idx] - feature_means
        else:
            feature_means = np.mean(data[idx], axis=0)
            data[idx] = data[idx] - feature_means[None, :]
    return data