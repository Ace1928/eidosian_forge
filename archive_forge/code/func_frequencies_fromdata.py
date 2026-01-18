import warnings
import numpy as np
from scipy import interpolate, stats
def frequencies_fromdata(data, k_bins, use_ranks=True):
    """count of observations in bins (histogram)

    currently only for bivariate data

    Parameters
    ----------
    data : array_like
        Bivariate data with observations in rows and two columns. Binning is
        in unit rectangle [0, 1]^2. If use_rank is False, then data should be
        in unit interval.
    k_bins : int
        Number of bins along each dimension in the histogram
    use_ranks : bool
        If use_rank is True, then data will be converted to ranks without
        tie handling.

    Returns
    -------
    bin counts : ndarray
        Frequencies are the number of observations in a given bin.
        Bin counts are a 2-dim array with k_bins rows and k_bins columns.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.
    """
    data = np.asarray(data)
    k_dim = data.shape[-1]
    k = k_bins + 1
    g2 = _Grid([k] * k_dim, eps=0)
    if use_ranks:
        data = _rankdata_no_ties(data) / (data.shape[0] + 1)
    freqr, _ = np.histogramdd(data, bins=g2.x_marginal)
    return freqr