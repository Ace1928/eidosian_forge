from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def power_proportions_2indep(diff, prop2, nobs1, ratio=1, alpha=0.05, value=0, alternative='two-sided', return_results=True):
    """
    Power for ztest that two independent proportions are equal

    This assumes that the variance is based on the pooled proportion
    under the null and the non-pooled variance under the alternative

    Parameters
    ----------
    diff : float
        difference between proportion 1 and 2 under the alternative
    prop2 : float
        proportion for the reference case, prop2, proportions for the
        first case will be computed using p2 and diff
        p1 = p2 + diff
    nobs1 : float or int
        number of observations in sample 1
    ratio : float
        sample size ratio, nobs2 = ratio * nobs1
    alpha : float in interval (0,1)
        Significance level, e.g. 0.05, is the probability of a type I
        error, that is wrong rejections if the Null Hypothesis is true.
    value : float
        currently only `value=0`, i.e. equality testing, is supported
    alternative : string, 'two-sided' (default), 'larger', 'smaller'
        Alternative hypothesis whether the power is calculated for a
        two-sided (default) or one sided test. The one-sided test can be
        either 'larger', 'smaller'.
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise only the computed power is returned.

    Returns
    -------
    results : results instance or float
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only the power is returned.

        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Other attributes in results instance include :

        p_pooled
            pooled proportion, used for std_null
        std_null
            standard error of difference under the null hypothesis (without
            sqrt(nobs1))
        std_alt
            standard error of difference under the alternative hypothesis
            (without sqrt(nobs1))
    """
    from statsmodels.stats.power import normal_power_het
    p_pooled, std_null, std_alt = _std_2prop_power(diff, prop2, ratio=ratio, alpha=alpha, value=value)
    pow_ = normal_power_het(diff, nobs1, alpha, std_null=std_null, std_alternative=std_alt, alternative=alternative)
    if return_results:
        res = Holder(power=pow_, p_pooled=p_pooled, std_null=std_null, std_alt=std_alt, nobs1=nobs1, nobs2=ratio * nobs1, nobs_ratio=ratio, alpha=alpha)
        return res
    else:
        return pow_