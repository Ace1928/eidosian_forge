from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def highly_variable_genes(data, *extra_data, kernel_size=0.05, smooth=5, cutoff=None, percentile=80):
    """Select genes with high variability.

    Variability is computed as the deviation from a loess fit
    to the rolling median of the mean-variance curve

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same rows
    kernel_size : float or int, optional (default: 0.005)
        Width of rolling median window. If a float between 0 and 1, the width is given
        by kernel_size * data.shape[1]. Otherwise should be an odd integer
    smooth : int, optional (default: 5)
        Amount of smoothing to apply to the median filter
    cutoff : float, optional (default: None)
        Variability above which expression is deemed significant
    percentile : int, optional (Default: 80)
        Percentile above or below which to remove genes.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    extra_data : array-like, shape=[any, m_features]
        Filtered extra data, if passed.
    """
    from . import measure
    var_genes = measure.gene_variability(data, kernel_size=kernel_size, smooth=smooth)
    keep_cells_idx = utils._get_filter_idx(var_genes, cutoff, percentile, keep_cells='above')
    return select_cols(data, *extra_data, idx=keep_cells_idx)