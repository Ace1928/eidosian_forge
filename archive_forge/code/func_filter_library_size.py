from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def filter_library_size(data, *extra_data, cutoff=None, percentile=None, keep_cells=None, return_library_size=False, sample_labels=None, filter_per_sample=None):
    """Remove all cells with library size above or below a certain threshold.

    It is recommended to use :func:`~scprep.plot.plot_library_size` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    cutoff : float or tuple of floats, optional (default: None)
        Library size above or below which to retain a cell. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain a cell.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above, below or between the cutoff.
        If None, defaults to 'above' when a single cutoff is given and
        'between' when two cutoffs are given.
    return_library_size : bool, optional (default: False)
        If True, also return the library sizes corresponding to the retained cells
    sample_labels : Deprecated
    filter_per_sample : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    filtered_library_size : list-like, shape=[m_samples]
        Library sizes corresponding to retained samples,
        returned only if return_library_size is True
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    cell_sums = measure.library_size(data)
    return filter_values(data, *extra_data, values=cell_sums, cutoff=cutoff, percentile=percentile, keep_cells=keep_cells, return_values=return_library_size, sample_labels=sample_labels, filter_per_sample=filter_per_sample)