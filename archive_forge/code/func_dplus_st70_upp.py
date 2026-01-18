from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def dplus_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = np.exp(-2 * stat_modified ** 2)
    digits = np.sum(stat > np.array([0.82, 0.82, 1.0]))
    return (stat_modified, pval, digits)