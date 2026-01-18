import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def symmetry_bowker(table):
    """Test for symmetry of a (k, k) square contingency table

    This is an extension of the McNemar test to test the Null hypothesis
    that the contingency table is symmetric around the main diagonal, that is

    n_{i, j} = n_{j, i}  for all i, j

    Parameters
    ----------
    table : array_like, 2d, (k, k)
        a square contingency table that contains the count for k categories
        in rows and columns.

    Returns
    -------
    statistic : float
        chisquare test statistic
    p-value : float
        p-value of the test statistic based on chisquare distribution
    df : int
        degrees of freedom of the chisquare distribution

    Notes
    -----
    Implementation is based on the SAS documentation, R includes it in
    `mcnemar.test` if the table is not 2 by 2.

    The pvalue is based on the chisquare distribution which requires that the
    sample size is not very small to be a good approximation of the true
    distribution. For 2x2 contingency tables exact distribution can be
    obtained with `mcnemar`

    See Also
    --------
    mcnemar


    """
    warnings.warn('Deprecated, use stats.TableSymmetry instead', FutureWarning)
    table = np.asarray(table)
    k, k2 = table.shape
    if k != k2:
        raise ValueError('table needs to be square')
    upp_idx = np.triu_indices(k, 1)
    tril = table.T[upp_idx]
    triu = table[upp_idx]
    stat = ((tril - triu) ** 2 / (tril + triu + 1e-20)).sum()
    df = k * (k - 1) / 2.0
    pval = stats.chi2.sf(stat, df)
    return (stat, pval, df)