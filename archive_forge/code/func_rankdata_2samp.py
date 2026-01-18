import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def rankdata_2samp(x1, x2):
    """Compute midranks for two samples

    Parameters
    ----------
    x1, x2 : array_like
        Original data for two samples that will be converted to midranks.

    Returns
    -------
    rank1 : ndarray
        Midranks of the first sample in the pooled sample.
    rank2 : ndarray
        Midranks of the second sample in the pooled sample.
    ranki1 : ndarray
        Internal midranks of the first sample.
    ranki2 : ndarray
        Internal midranks of the second sample.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    nobs1 = len(x1)
    nobs2 = len(x2)
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError('one sample has zero length')
    x_combined = np.concatenate((x1, x2))
    if x_combined.ndim > 1:
        rank = np.apply_along_axis(rankdata, 0, x_combined)
    else:
        rank = rankdata(x_combined)
    rank1 = rank[:nobs1]
    rank2 = rank[nobs1:]
    if x_combined.ndim > 1:
        ranki1 = np.apply_along_axis(rankdata, 0, x1)
        ranki2 = np.apply_along_axis(rankdata, 0, x2)
    else:
        ranki1 = rankdata(x1)
        ranki2 = rankdata(x2)
    return (rank1, rank2, ranki1, ranki2)