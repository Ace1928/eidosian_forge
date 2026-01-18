from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def distance_covariance_test(x, y, B=None, method='auto'):
    """The Distance Covariance (dCov) test

    Apply the Distance Covariance (dCov) test of independence to `x` and `y`.
    This test was introduced in [1]_, and is based on the distance covariance
    statistic. The test is applicable to random vectors of arbitrary length
    (see the notes section for more details).

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    B : int, optional, default=`None`
        The number of iterations to perform when evaluating the null
        distribution of the test statistic when the `emp` method is
        applied (see below). if `B` is `None` than as in [1]_ we set
        `B` to be ``B = 200 + 5000/n``, where `n` is the number of
        observations.
    method : {'auto', 'emp', 'asym'}, optional, default=auto
        The method by which to obtain the p-value for the test.

        - `auto` : Default method. The number of observations will be used to
          determine the method.
        - `emp` : Empirical evaluation of the p-value using permutations of
          the rows of `y` to obtain the null distribution.
        - `asym` : An asymptotic approximation of the distribution of the test
          statistic is used to find the p-value.

    Returns
    -------
    test_statistic : float
        The value of the test statistic used in the test.
    pval : float
        The p-value.
    chosen_method : str
        The method that was used to obtain the p-value. Mostly relevant when
        the function is called with `method='auto'`.

    Notes
    -----
    The test applies to random vectors of arbitrary dimensions, i.e., `x`
    can be a 1-D vector of observations for a single random variable while
    `y` can be a `k` by `n` 2-D array (where `k > 1`). In other words, it
    is also possible for `x` and `y` to both be 2-D arrays and have the
    same number of rows (observations) while differing in the number of
    columns.

    As noted in [1]_ the statistics are sensitive to all types of departures
    from independence, including nonlinear or nonmonotone dependence
    structure.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------
    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_covariance_test
    >>> data = np.random.rand(1000, 10)
    >>> x, y = data[:, :3], data[:, 3:]
    >>> x.shape
    (1000, 3)
    >>> y.shape
    (1000, 7)
    >>> distance_covariance_test(x, y)
    (1.0426404792714983, 0.2971148340813543, 'asym')
    # (test_statistic, pval, chosen_method)

    """
    x, y = _validate_and_tranform_x_and_y(x, y)
    n = x.shape[0]
    stats = distance_statistics(x, y)
    if method == 'auto' and n <= 500 or method == 'emp':
        chosen_method = 'emp'
        test_statistic, pval = _empirical_pvalue(x, y, B, n, stats)
    elif method == 'auto' and n > 500 or method == 'asym':
        chosen_method = 'asym'
        test_statistic, pval = _asymptotic_pvalue(stats)
    else:
        raise ValueError(f"Unknown 'method' parameter: {method}")
    if chosen_method == 'emp' and pval in [0, 1]:
        msg = f'p-value was {pval} when using the empirical method. The asymptotic approximation will be used instead'
        warnings.warn(msg, HypothesisTestWarning)
        _, pval = _asymptotic_pvalue(stats)
    return (test_statistic, pval, chosen_method)