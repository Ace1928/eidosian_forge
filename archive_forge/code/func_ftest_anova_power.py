import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ftest_anova_power(effect_size, nobs, alpha, k_groups=2, df=None):
    """power for ftest for one way anova with k equal sized groups

    nobs total sample size, sum over all groups

    should be general nobs observations, k_groups restrictions ???
    """
    df_num = k_groups - 1
    df_denom = nobs - k_groups
    crit = stats.f.isf(alpha, df_num, df_denom)
    pow_ = ncf_sf(crit, df_num, df_denom, effect_size ** 2 * nobs)
    return pow_