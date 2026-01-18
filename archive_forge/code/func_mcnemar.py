import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def mcnemar(x, y=None, exact=True, correction=True):
    """McNemar test

    Parameters
    ----------
    x, y : array_like
        two paired data samples. If y is None, then x can be a 2 by 2
        contingency table. x and y can have more than one dimension, then
        the results are calculated under the assumption that axis zero
        contains the observation for the samples.
    exact : bool
        If exact is true, then the binomial distribution will be used.
        If exact is false, then the chisquare distribution will be used, which
        is the approximation to the distribution of the test statistic for
        large sample sizes.
    correction : bool
        If true, then a continuity correction is used for the chisquare
        distribution (if exact is false.)

    Returns
    -------
    stat : float or int, array
        The test statistic is the chisquare statistic if exact is false. If the
        exact binomial distribution is used, then this contains the min(n1, n2),
        where n1, n2 are cases that are zero in one sample but one in the other
        sample.

    pvalue : float or array
        p-value of the null hypothesis of equal effects.

    Notes
    -----
    This is a special case of Cochran's Q test. The results when the chisquare
    distribution is used are identical, except for continuity correction.

    """
    warnings.warn('Deprecated, use stats.TableSymmetry instead', FutureWarning)
    x = np.asarray(x)
    if y is None and x.shape[0] == x.shape[1]:
        if x.shape[0] != 2:
            raise ValueError('table needs to be 2 by 2')
        n1, n2 = (x[1, 0], x[0, 1])
    else:
        n1 = np.sum(x < y, 0)
        n2 = np.sum(x > y, 0)
    if exact:
        stat = np.minimum(n1, n2)
        pval = stats.binom.cdf(stat, n1 + n2, 0.5) * 2
        pval = np.minimum(pval, 1)
    else:
        corr = int(correction)
        stat = (np.abs(n1 - n2) - corr) ** 2 / (1.0 * (n1 + n2))
        df = 1
        pval = stats.chi2.sf(stat, df)
    return (stat, pval)