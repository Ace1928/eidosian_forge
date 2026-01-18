import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like
def correlation_sums(indicators, max_dim):
    """
    Calculate all correlation sums for embedding dimensions 1:max_dim

    Parameters
    ----------
    indicators : 2d array
        matrix of distance threshold indicators
    max_dim : int
        maximum embedding dimension

    Returns
    -------
    corrsums : ndarray
        Correlation sums
    """
    corrsums = np.zeros((1, max_dim))
    corrsums[0, 0], indicators = correlation_sum(indicators, 1)
    for i in range(1, max_dim):
        corrsums[0, i], indicators = correlation_sum(indicators, 2)
    return corrsums