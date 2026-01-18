import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like
def bds(x, max_dim=2, epsilon=None, distance=1.5):
    """
    BDS Test Statistic for Independence of a Time Series

    Parameters
    ----------
    x : ndarray
        Observations of time series for which bds statistics is calculated.
    max_dim : int
        The maximum embedding dimension.
    epsilon : {float, None}, optional
        The threshold distance to use in calculating the correlation sum.
    distance : float, optional
        Specifies the distance multiplier to use when computing the test
        statistic if epsilon is omitted.

    Returns
    -------
    bds_stat : float
        The BDS statistic.
    pvalue : float
        The p-values associated with the BDS statistic.

    Notes
    -----
    The null hypothesis of the test statistic is for an independent and
    identically distributed (i.i.d.) time series, and an unspecified
    alternative hypothesis.

    This test is often used as a residual diagnostic.

    The calculation involves matrices of size (nobs, nobs), so this test
    will not work with very long datasets.

    Implementation conditions on the first m-1 initial values, which are
    required to calculate the m-histories:
    x_t^m = (x_t, x_{t-1}, ... x_{t-(m-1)})
    """
    x = array_like(x, 'x', ndim=1)
    nobs_full = len(x)
    if max_dim < 2 or max_dim >= nobs_full:
        raise ValueError('Maximum embedding dimension must be in the range [2,len(x)-1]. Got %d.' % max_dim)
    indicators = distance_indicators(x, epsilon, distance)
    corrsum_mdims = correlation_sums(indicators, max_dim)
    variances, k = _var(indicators, max_dim)
    stddevs = np.sqrt(variances)
    bds_stats = np.zeros((1, max_dim - 1))
    pvalues = np.zeros((1, max_dim - 1))
    for embedding_dim in range(2, max_dim + 1):
        ninitial = embedding_dim - 1
        nobs = nobs_full - ninitial
        corrsum_1dim, _ = correlation_sum(indicators[ninitial:, ninitial:], 1)
        corrsum_mdim = corrsum_mdims[0, embedding_dim - 1]
        effect = corrsum_mdim - corrsum_1dim ** embedding_dim
        sd = stddevs[0, embedding_dim - 2]
        bds_stats[0, embedding_dim - 2] = np.sqrt(nobs) * effect / sd
        pvalue = 2 * stats.norm.sf(np.abs(bds_stats[0, embedding_dim - 2]))
        pvalues[0, embedding_dim - 2] = pvalue
    return (np.squeeze(bds_stats), np.squeeze(pvalues))