from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
def chisquare_power(effect_size, nobs, n_bins, alpha=0.05, ddof=0):
    """power of chisquare goodness of fit test

    effect size is sqrt of chisquare statistic divided by nobs

    Parameters
    ----------
    effect_size : float
        This is the deviation from the Null of the normalized chi_square
        statistic. This follows Cohen's definition (sqrt).
    nobs : int or float
        number of observations
    n_bins : int (or float)
        number of bins, or points in the discrete distribution
    alpha : float in (0,1)
        significance level of the test, default alpha=0.05

    Returns
    -------
    power : float
        power of the test at given significance level at effect size

    Notes
    -----
    This function also works vectorized if all arguments broadcast.

    This can also be used to calculate the power for power divergence test.
    However, for the range of more extreme values of the power divergence
    parameter, this power is not a very good approximation for samples of
    small to medium size (Drost et al. 1989)

    References
    ----------
    Drost, ...

    See Also
    --------
    chisquare_effectsize
    statsmodels.stats.GofChisquarePower

    """
    crit = stats.chi2.isf(alpha, n_bins - 1 - ddof)
    power = stats.ncx2.sf(crit, n_bins - 1 - ddof, effect_size ** 2 * nobs)
    return power