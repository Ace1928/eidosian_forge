from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def filter_duplicates(data, *extra_data, sample_labels=None):
    """Filter all duplicate cells.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    sample_labels : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if sample_labels is not None:
        warnings.warn('`sample_labels` is deprecated. Passing `sample_labels` as `extra_data`.', DeprecationWarning)
        extra_data = list(extra_data) + [sample_labels]
    unique_idx = _find_unique_cells(data)
    data = select.select_rows(data, *extra_data, idx=unique_idx)
    return data