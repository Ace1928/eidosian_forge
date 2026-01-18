import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal
def movorder(x, order='med', windsize=3, lag='lagged'):
    """moving order statistics

    Parameters
    ----------
    x : ndarray
       time series data
    order : float or 'med', 'min', 'max'
       which order statistic to calculate
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    filtered array


    """
    if lag == 'lagged':
        lead = windsize // 2
    elif lag == 'centered':
        lead = 0
    elif lag == 'leading':
        lead = -windsize // 2 + 1
    else:
        raise ValueError
    if np.isfinite(order):
        ord = order
    elif order == 'med':
        ord = (windsize - 1) / 2
    elif order == 'min':
        ord = 0
    elif order == 'max':
        ord = windsize - 1
    else:
        raise ValueError
    xext = expandarr(x, windsize)
    return signal.order_filter(xext, np.ones(windsize), ord)[windsize - lead:-(windsize + lead)]