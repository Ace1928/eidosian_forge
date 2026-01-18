import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def normal_power_het(diff, nobs, alpha, std_null=1.0, std_alternative=None, alternative='two-sided'):
    """Calculate power of a normal distributed test statistic

    This is an generalization of `normal_power` when variance under Null and
    Alternative differ.

    Parameters
    ----------
    diff : float
        difference in the estimated means or statistics under the alternative.
    nobs : float or int
        number of observations
    alpha : float in interval (0,1)
        significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    std_null : float
        standard deviation under the Null hypothesis without division by
        sqrt(nobs)
    std_alternative : float
        standard deviation under the Alternative hypothesis without division
        by sqrt(nobs)
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        extra argument to choose whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.

    Returns
    -------
    power : float
    """
    d = diff
    if std_alternative is None:
        std_alternative = std_null
    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.0
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " + "or 'smaller'")
    std_ratio = std_null / std_alternative
    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit = stats.norm.isf(alpha_)
        pow_ = stats.norm.sf(crit * std_ratio - d * np.sqrt(nobs) / std_alternative)
    if alternative in ['two-sided', '2s', 'smaller']:
        crit = stats.norm.ppf(alpha_)
        pow_ += stats.norm.cdf(crit * std_ratio - d * np.sqrt(nobs) / std_alternative)
    return pow_