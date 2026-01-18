from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def filter_values(data, *extra_data, values=None, cutoff=None, percentile=None, keep_cells='above', return_values=False, sample_labels=None, filter_per_sample=None):
    """Remove all cells with `values` above or below a certain threshold.

    It is recommended to use :func:`~scprep.plot.histogram` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    values : list-like, shape=[n_samples]
        Value upon which to filter
    cutoff : float or tuple of floats, optional (default: None)
        Value above or below which to retain cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above, below or between the cutoff.
        If None, defaults to 'above' when a single cutoff is given and
        'between' when two cutoffs are given.
    return_values : bool, optional (default: False)
        If True, also return the values corresponding to the retained cells
    sample_labels : Deprecated
    filter_per_sample : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    filtered_values : list-like, shape=[m_samples]
        Values corresponding to retained samples,
        returned only if return_values is True
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if sample_labels is not None:
        warnings.warn('`sample_labels` is deprecated. Passing `sample_labels` as `extra_data`.', DeprecationWarning)
        extra_data = list(extra_data) + [sample_labels]
    if filter_per_sample is not None:
        warnings.warn('`filter_per_sample` is deprecated. Filtering as a single sample.', DeprecationWarning)
    assert values is not None
    keep_cells_idx = utils._get_filter_idx(values, cutoff, percentile, keep_cells)
    if return_values:
        extra_data = [values] + list(extra_data)
    data = select.select_rows(data, *extra_data, idx=keep_cells_idx)
    return data