from statsmodels.compat.python import lrange
from pprint import pprint
import numpy as np
def randintw(w, size=1):
    """generate integer random variables given probabilties

    useful because it can be used as index into any array or sequence type

    Parameters
    ----------
    w : 1d array_like
        sequence of weights, probabilities. The weights are normalized to add
        to one.
    size : int or tuple of ints
        shape of output array

    Returns
    -------
    rvs : array of shape given by size
        random variables each distributed according to the same discrete
        distribution defined by (normalized) w.

    Examples
    --------
    >>> np.random.seed(0)
    >>> randintw([0.4, 0.4, 0.2], size=(2,6))
    array([[1, 1, 1, 1, 1, 1],
           [1, 2, 2, 0, 1, 1]])

    >>> np.bincount(randintw([0.6, 0.4, 0.0], size=3000))/3000.
    array([ 0.59566667,  0.40433333])

    """
    from numpy.random import random
    p = np.cumsum(w) / np.sum(w)
    rvs = p.searchsorted(random(np.prod(size))).reshape(size)
    return rvs