from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
def omni_normtest(resids, axis=0):
    """
    Omnibus test for normality

    Parameters
    ----------
    resid : array_like
    axis : int, optional
        Default is 0

    Returns
    -------
    Chi^2 score, two-tail probability
    """
    resids = np.asarray(resids)
    n = resids.shape[axis]
    if n < 8:
        from warnings import warn
        warn('omni_normtest is not valid with less than 8 observations; %i samples were given.' % int(n), ValueWarning)
        return (np.nan, np.nan)
    return stats.normaltest(resids, axis=axis)