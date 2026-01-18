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
def proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False):
    """
    Test for proportions based on normal (z) test

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : {int, array_like}
        the number of trials or observations, with the same length as
        count.
    value : float, array_like or None, optional
        This is the value of the null hypothesis equal to the proportion in the
        case of a one sample test. In the case of a two-sample test, the
        null hypothesis is that prop[0] - prop[1] = value, where prop is the
        proportion in the two samples. If not provided value = 0 and the null
        is prop[0] = prop[1]
    alternative : str in ['two-sided', 'smaller', 'larger']
        The alternative hypothesis can be either two-sided or one of the one-
        sided tests, smaller means that the alternative hypothesis is
        ``prop < value`` and larger means ``prop > value``. In the two sample
        test, smaller means that the alternative hypothesis is ``p1 < p2`` and
        larger means ``p1 > p2`` where ``p1`` is the proportion of the first
        sample and ``p2`` of the second one.
    prop_var : False or float in (0, 1)
        If prop_var is false, then the variance of the proportion estimate is
        calculated based on the sample proportion. Alternatively, a proportion
        can be specified to calculate this variance. Common use case is to
        use the proportion under the Null hypothesis to specify the variance
        of the proportion estimate.

    Returns
    -------
    zstat : float
        test statistic for the z-test
    p-value : float
        p-value for the z-test

    Examples
    --------
    >>> count = 5
    >>> nobs = 83
    >>> value = .05
    >>> stat, pval = proportions_ztest(count, nobs, value)
    >>> print('{0:0.3f}'.format(pval))
    0.695

    >>> import numpy as np
    >>> from statsmodels.stats.proportion import proportions_ztest
    >>> count = np.array([5, 12])
    >>> nobs = np.array([83, 99])
    >>> stat, pval = proportions_ztest(count, nobs)
    >>> print('{0:0.3f}'.format(pval))
    0.159

    Notes
    -----
    This uses a simple normal test for proportions. It should be the same as
    running the mean z-test on the data encoded 1 for event and 0 for no event
    so that the sum corresponds to the count.

    In the one and two sample cases with two-sided alternative, this test
    produces the same p-value as ``proportions_chisquare``, since the
    chisquare is the distribution of the square of a standard normal
    distribution.
    """
    count = np.asarray(count)
    nobs = np.asarray(nobs)
    if nobs.size == 1:
        nobs = nobs * np.ones_like(count)
    prop = count * 1.0 / nobs
    k_sample = np.size(prop)
    if value is None:
        if k_sample == 1:
            raise ValueError('value must be provided for a 1-sample test')
        value = 0
    if k_sample == 1:
        diff = prop - value
    elif k_sample == 2:
        diff = prop[0] - prop[1] - value
    else:
        msg = 'more than two samples are not implemented yet'
        raise NotImplementedError(msg)
    p_pooled = np.sum(count) * 1.0 / np.sum(nobs)
    nobs_fact = np.sum(1.0 / nobs)
    if prop_var:
        p_pooled = prop_var
    var_ = p_pooled * (1 - p_pooled) * nobs_fact
    std_diff = np.sqrt(var_)
    from statsmodels.stats.weightstats import _zstat_generic2
    return _zstat_generic2(diff, std_diff, alternative)