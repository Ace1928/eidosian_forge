import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def confint_poisson_2indep(count1, exposure1, count2, exposure2, method='score', compare='ratio', alpha=0.05, method_mover='score'):
    """Confidence interval for ratio or difference of 2 indep poisson rates.

    Parameters
    ----------
    count1 : int
        Number of events in first sample.
    exposure1 : float
        Total exposure (time * subjects) in first sample.
    count2 : int
        Number of events in second sample.
    exposure2 : float
        Total exposure (time * subjects) in second sample.
    method : string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        see Notes.

        ratio:

        - 'wald': NOT YET, method W1A, wald test, variance based on observed
          rates
        - 'waldcc' :
        - 'score': method W2A, score test, variance based on estimate under
          the Null hypothesis
        - 'wald-log': W3A, uses log-ratio, variance based on observed rates
        - 'score-log' W4A, uses log-ratio, variance based on estimate under
          the Null hypothesis
        - 'sqrt': W5A, based on variance stabilizing square root transformation
        - 'sqrtcc' :
        - 'exact-cond': NOT YET, exact conditional test based on binomial
          distribution
          This uses ``binom_test`` which is minlike in the two-sided case.
        - 'cond-midp': NOT YET, midpoint-pvalue of exact conditional test
        - 'mover' :

        diff:

        - 'wald',
        - 'waldccv'
        - 'score'
        - 'mover'

    compare : {'diff', 'ratio'}
        Default is "ratio".
        If compare is `diff`, then the hypothesis test is for
        diff = rate1 - rate2.
        If compare is `ratio`, then the hypothesis test is for the
        rate ratio defined by ratio = rate1 / rate2.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio of rates is not equal to ratio_null (default)
        - 'larger' :   H1: ratio of rates is larger than ratio_null
        - 'smaller' :  H1: ratio of rates is smaller than ratio_null

    alpha : float in (0, 1)
        Significance level, nominal coverage of the confidence interval is
        1 - alpha.

    Returns
    -------
    tuple (low, upp) : confidence limits.

    """
    y1, n1, y2, n2 = map(np.asarray, [count1, exposure1, count2, exposure2])
    rate1, rate2 = (y1 / n1, y2 / n2)
    alpha = alpha / 2
    if compare == 'ratio':
        if method == 'score':
            low, upp = _invert_test_confint_2indep(count1, exposure1, count2, exposure2, alpha=alpha * 2, method='score', compare='ratio', method_start='waldcc')
            ci = (low, upp)
        elif method == 'wald-log':
            crit = stats.norm.isf(alpha)
            c = 0
            center = (count1 + c) / (count2 + c) * n2 / n1
            std = np.sqrt(1 / (count1 + c) + 1 / (count2 + c))
            ci = (center * np.exp(-crit * std), center * np.exp(crit * std))
        elif method == 'score-log':
            low, upp = _invert_test_confint_2indep(count1, exposure1, count2, exposure2, alpha=alpha * 2, method='score-log', compare='ratio', method_start='waldcc')
            ci = (low, upp)
        elif method == 'waldcc':
            crit = stats.norm.isf(alpha)
            center = (count1 + 0.5) / (count2 + 0.5) * n2 / n1
            std = np.sqrt(1 / (count1 + 0.5) + 1 / (count2 + 0.5))
            ci = (center * np.exp(-crit * std), center * np.exp(crit * std))
        elif method == 'sqrtcc':
            crit = stats.norm.isf(alpha)
            center = np.sqrt((count1 + 0.5) * (count2 + 0.5))
            std = 0.5 * np.sqrt(count1 + 0.5 + count2 + 0.5 - 0.25 * crit)
            denom = count2 + 0.5 - 0.25 * crit ** 2
            low_sqrt = (center - crit * std) / denom
            upp_sqrt = (center + crit * std) / denom
            ci = (low_sqrt ** 2, upp_sqrt ** 2)
        elif method == 'mover':
            method_p = method_mover
            ci1 = confint_poisson(y1, n1, method=method_p, alpha=2 * alpha)
            ci2 = confint_poisson(y2, n2, method=method_p, alpha=2 * alpha)
            ci = _mover_confint(rate1, rate2, ci1, ci2, contrast='ratio')
        else:
            raise ValueError(f'method "{method}" not recognized')
        ci = (np.maximum(ci[0], 0), ci[1])
    elif compare == 'diff':
        if method in ['wald']:
            crit = stats.norm.isf(alpha)
            center = rate1 - rate2
            half = crit * np.sqrt(rate1 / n1 + rate2 / n2)
            ci = (center - half, center + half)
        elif method in ['waldccv']:
            crit = stats.norm.isf(alpha)
            center = rate1 - rate2
            std = np.sqrt((count1 + 0.5) / n1 ** 2 + (count2 + 0.5) / n2 ** 2)
            half = crit * std
            ci = (center - half, center + half)
        elif method == 'score':
            low, upp = _invert_test_confint_2indep(count1, exposure1, count2, exposure2, alpha=alpha * 2, method='score', compare='diff', method_start='waldccv')
            ci = (low, upp)
        elif method == 'mover':
            method_p = method_mover
            ci1 = confint_poisson(y1, n1, method=method_p, alpha=2 * alpha)
            ci2 = confint_poisson(y2, n2, method=method_p, alpha=2 * alpha)
            ci = _mover_confint(rate1, rate2, ci1, ci2, contrast='diff')
        else:
            raise ValueError(f'method "{method}" not recognized')
    else:
        raise NotImplementedError('"compare" needs to be ratio or diff')
    return ci