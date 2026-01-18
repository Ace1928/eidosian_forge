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
def score_test_proportions_2indep(count1, nobs1, count2, nobs2, value=None, compare='diff', alternative='two-sided', correction=True, return_results=True):
    """
    Score test for two independent proportions

    This uses the constrained estimate of the proportions to compute
    the variance under the Null hypothesis.

    Parameters
    ----------
    count1, nobs1 :
        count and sample size for first sample
    count2, nobs2 :
        count and sample size for the second sample
    value : float
        diff, ratio or odds-ratio under the null hypothesis. If value is None,
        then equality of proportions under the Null is assumed,
        i.e. value=0 for 'diff' or value=1 for either rate or odds-ratio.
    compare : string in ['diff', 'ratio' 'odds-ratio']
        If compare is diff, then the confidence interval is for diff = p1 - p2.
        If compare is ratio, then the confidence interval is for the risk ratio
        defined by ratio = p1 / p2.
        If compare is odds-ratio, then the confidence interval is for the
        odds-ratio defined by or = p1 / (1 - p1) / (p2 / (1 - p2)
    return_results : bool
        If true, then a results instance with extra information is returned,
        otherwise a tuple with statistic and pvalue is returned.

    Returns
    -------
    results : results instance or tuple
        If return_results is True, then a results instance with the
        information in attributes is returned.
        If return_results is False, then only ``statistic`` and ``pvalue``
        are returned.

        statistic : float
            test statistic asymptotically normal distributed N(0, 1)
        pvalue : float
            p-value based on normal distribution
        other attributes :
            additional information about the hypothesis test

    Notes
    -----
    Status: experimental, the type or extra information in the return might
    change.

    """
    value_default = 0 if compare == 'diff' else 1
    if value is None:
        value = value_default
    nobs = nobs1 + nobs2
    count = count1 + count2
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    if value == value_default:
        prop0 = prop1 = count / nobs
    count0, nobs0 = (count2, nobs2)
    p0 = p2
    if compare == 'diff':
        diff = value
        if diff != 0:
            tmp3 = nobs
            tmp2 = (nobs1 + 2 * nobs0) * diff - nobs - count
            tmp1 = (count0 * diff - nobs - 2 * count0) * diff + count
            tmp0 = count0 * diff * (1 - diff)
            q = (tmp2 / (3 * tmp3)) ** 3 - tmp1 * tmp2 / (6 * tmp3 ** 2) + tmp0 / (2 * tmp3)
            p = np.sign(q) * np.sqrt((tmp2 / (3 * tmp3)) ** 2 - tmp1 / (3 * tmp3))
            a = (np.pi + np.arccos(q / p ** 3)) / 3
            prop0 = 2 * p * np.cos(a) - tmp2 / (3 * tmp3)
            prop1 = prop0 + diff
        var = prop1 * (1 - prop1) / nobs1 + prop0 * (1 - prop0) / nobs0
        if correction:
            var *= nobs / (nobs - 1)
        diff_stat = p1 - p0 - diff
    elif compare == 'ratio':
        ratio = value
        if ratio != 1:
            a = nobs * ratio
            b = -(nobs1 * ratio + count1 + nobs2 + count0 * ratio)
            c = count
            prop0 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            prop1 = prop0 * ratio
        var = prop1 * (1 - prop1) / nobs1 + ratio ** 2 * prop0 * (1 - prop0) / nobs0
        if correction:
            var *= nobs / (nobs - 1)
        diff_stat = p1 - ratio * p0
    elif compare in ['or', 'odds-ratio']:
        oratio = value
        if oratio != 1:
            a = nobs0 * (oratio - 1)
            b = nobs1 * oratio + nobs0 - count * (oratio - 1)
            c = -count
            prop0 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            prop1 = prop0 * oratio / (1 + prop0 * (oratio - 1))
        eps = 1e-10
        prop0 = np.clip(prop0, eps, 1 - eps)
        prop1 = np.clip(prop1, eps, 1 - eps)
        var = 1 / (prop1 * (1 - prop1) * nobs1) + 1 / (prop0 * (1 - prop0) * nobs0)
        if correction:
            var *= nobs / (nobs - 1)
        diff_stat = (p1 - prop1) / (prop1 * (1 - prop1)) - (p0 - prop0) / (prop0 * (1 - prop0))
    statistic, pvalue = _zstat_generic2(diff_stat, np.sqrt(var), alternative=alternative)
    if return_results:
        res = HolderTuple(statistic=statistic, pvalue=pvalue, compare=compare, method='score', variance=var, alternative=alternative, prop1_null=prop1, prop2_null=prop0)
        return res
    else:
        return (statistic, pvalue)