import warnings
import numpy as np
from scipy import interpolate, stats
def prob2cdf_grid(probs):
    """Cumulative probabilities from cell provabilites on a grid

    Parameters
    ----------
    probs : array_like
        Rectangular grid of cell probabilities.

    Returns
    -------
    cdf : ndarray
        Grid of cumulative probabilities with same shape as probs.
    """
    cdf = np.asarray(probs).copy()
    k = cdf.ndim
    for i in range(k):
        cdf = cdf.cumsum(axis=i)
    return cdf