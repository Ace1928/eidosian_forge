import warnings
import numpy as np
from scipy import interpolate, stats
def cdf2prob_grid(cdf, prepend=0):
    """Cell probabilities from cumulative probabilities on a grid.

    Parameters
    ----------
    cdf : array_like
        Grid of cumulative probabilities with same shape as probs.

    Returns
    -------
    probs : ndarray
        Rectangular grid of cell probabilities.

    """
    if prepend is None:
        prepend = np._NoValue
    prob = np.asarray(cdf).copy()
    k = prob.ndim
    for i in range(k):
        prob = np.diff(prob, prepend=prepend, axis=i)
    return prob