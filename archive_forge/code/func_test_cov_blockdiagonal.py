import numpy as np
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.base import HolderTuple
from statsmodels.tools.validation import array_like
def test_cov_blockdiagonal(cov, nobs, block_len):
    """One sample hypothesis test that covariance is block diagonal.

    The Null and alternative hypotheses are

    .. math::

       H0 &: \\Sigma = diag(\\Sigma_i) \\\\
       H1 &: \\Sigma \\neq diag(\\Sigma_i)

    where :math:`\\Sigma_i` are covariance blocks with unspecified values.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the data, estimated with denominator ``(N - 1)``,
        i.e. `ddof=1`.
    nobs : int
        number of observations used in the estimation of the covariance
    block_len : list
        list of length of each square block

    Returns
    -------
    res : instance of HolderTuple
        results with ``statistic, pvalue`` and other attributes like ``df``

    References
    ----------
    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.
    """
    cov = np.asarray(cov)
    cov_blocks = _get_blocks(cov, block_len)[0]
    k = cov.shape[0]
    k_blocks = [c.shape[0] for c in cov_blocks]
    if k != sum(k_blocks):
        msg = 'sample covariances and blocks do not have matching shape'
        raise ValueError(msg)
    logdet_blocks = sum((_logdet(c) for c in cov_blocks))
    a2 = k ** 2 - sum((ki ** 2 for ki in k_blocks))
    a3 = k ** 3 - sum((ki ** 3 for ki in k_blocks))
    statistic = nobs - 1 - (2 * a3 + 3 * a2) / (6.0 * a2)
    statistic *= logdet_blocks - _logdet(cov)
    df = a2 / 2
    pvalue = stats.chi2.sf(statistic, df)
    return HolderTuple(statistic=statistic, pvalue=pvalue, df=df, distr='chi2', null='block-diagonal')