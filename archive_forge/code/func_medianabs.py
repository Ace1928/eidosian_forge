import numpy as np
from statsmodels.tools.validation import array_like
def medianabs(x1, x2, axis=0):
    """median absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianabs : ndarray or float
       median absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(np.abs(x1 - x2), axis=axis)