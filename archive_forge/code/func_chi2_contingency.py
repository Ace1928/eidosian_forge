from functools import reduce
import math
import numpy as np
from ._stats_py import power_divergence
from ._relative_risk import relative_risk
from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from scipy._lib._bunch import _make_tuple_bunch
def chi2_contingency(observed, correction=True, lambda_=None):
    """Chi-square test of independence of variables in a contingency table.

    This function computes the chi-square statistic and p-value for the
    hypothesis test of independence of the observed frequencies in the
    contingency table [1]_ `observed`.  The expected frequencies are computed
    based on the marginal sums under the assumption of independence; see
    `scipy.stats.contingency.expected_freq`.  The number of degrees of
    freedom is (expressed using numpy functions and attributes)::

        dof = observed.size - sum(observed.shape) + observed.ndim - 1


    Parameters
    ----------
    observed : array_like
        The contingency table. The table contains the observed frequencies
        (i.e. number of occurrences) in each category.  In the two-dimensional
        case, the table is often described as an "R x C table".
    correction : bool, optional
        If True, *and* the degrees of freedom is 1, apply Yates' correction
        for continuity.  The effect of the correction is to adjust each
        observed value by 0.5 towards the corresponding expected value.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic [2]_.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family [3]_ to be used instead.  See
        `scipy.stats.power_divergence` for details.

    Returns
    -------
    res : Chi2ContingencyResult
        An object containing attributes:

        statistic : float
            The test statistic.
        pvalue : float
            The p-value of the test.
        dof : int
            The degrees of freedom.
        expected_freq : ndarray, same shape as `observed`
            The expected frequencies, based on the marginal sums of the table.

    See Also
    --------
    scipy.stats.contingency.expected_freq
    scipy.stats.fisher_exact
    scipy.stats.chisquare
    scipy.stats.power_divergence
    scipy.stats.barnard_exact
    scipy.stats.boschloo_exact

    Notes
    -----
    An often quoted guideline for the validity of this calculation is that
    the test should be used only if the observed and expected frequencies
    in each cell are at least 5.

    This is a test for the independence of different categories of a
    population. The test is only meaningful when the dimension of
    `observed` is two or more.  Applying the test to a one-dimensional
    table will always result in `expected` equal to `observed` and a
    chi-square statistic equal to 0.

    This function does not handle masked arrays, because the calculation
    does not make sense with missing values.

    Like `scipy.stats.chisquare`, this function computes a chi-square
    statistic; the convenience this function provides is to figure out the
    expected frequencies and degrees of freedom from the given contingency
    table. If these were already known, and if the Yates' correction was not
    required, one could use `scipy.stats.chisquare`.  That is, if one calls::

        res = chi2_contingency(obs, correction=False)

    then the following is true::

        (res.statistic, res.pvalue) == stats.chisquare(obs.ravel(),
                                                       f_exp=ex.ravel(),
                                                       ddof=obs.size - 1 - dof)

    The `lambda_` argument was added in version 0.13.0 of scipy.

    References
    ----------
    .. [1] "Contingency table",
           https://en.wikipedia.org/wiki/Contingency_table
    .. [2] "Pearson's chi-squared test",
           https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In [4]_, the use of aspirin to prevent cardiovascular events in women
    and men was investigated. The study notably concluded:

        ...aspirin therapy reduced the risk of a composite of
        cardiovascular events due to its effect on reducing the risk of
        ischemic stroke in women [...]

    The article lists studies of various cardiovascular events. Let's
    focus on the ischemic stoke in women.

    The following table summarizes the results of the experiment in which
    participants took aspirin or a placebo on a regular basis for several
    years. Cases of ischemic stroke were recorded::

                          Aspirin   Control/Placebo
        Ischemic stroke     176           230
        No stroke         21035         21018

    Is there evidence that the aspirin reduces the risk of ischemic stroke?
    We begin by formulating a null hypothesis :math:`H_0`:

        The effect of aspirin is equivalent to that of placebo.

    Let's assess the plausibility of this hypothesis with
    a chi-square test.

    >>> import numpy as np
    >>> from scipy.stats import chi2_contingency
    >>> table = np.array([[176, 230], [21035, 21018]])
    >>> res = chi2_contingency(table)
    >>> res.statistic
    6.892569132546561
    >>> res.pvalue
    0.008655478161175739

    Using a significance level of 5%, we would reject the null hypothesis in
    favor of the alternative hypothesis: "the effect of aspirin
    is not equivalent to the effect of placebo".
    Because `scipy.stats.contingency.chi2_contingency` performs a two-sided
    test, the alternative hypothesis does not indicate the direction of the
    effect. We can use `stats.contingency.odds_ratio` to support the
    conclusion that aspirin *reduces* the risk of ischemic stroke.

    Below are further examples showing how larger contingency tables can be
    tested.

    A two-way example (2 x 3):

    >>> obs = np.array([[10, 10, 20], [20, 20, 20]])
    >>> res = chi2_contingency(obs)
    >>> res.statistic
    2.7777777777777777
    >>> res.pvalue
    0.24935220877729619
    >>> res.dof
    2
    >>> res.expected_freq
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    Perform the test using the log-likelihood ratio (i.e. the "G-test")
    instead of Pearson's chi-squared statistic.

    >>> res = chi2_contingency(obs, lambda_="log-likelihood")
    >>> res.statistic
    2.7688587616781319
    >>> res.pvalue
    0.25046668010954165

    A four-way example (2 x 2 x 2 x 2):

    >>> obs = np.array(
    ...     [[[[12, 17],
    ...        [11, 16]],
    ...       [[11, 12],
    ...        [15, 16]]],
    ...      [[[23, 15],
    ...        [30, 22]],
    ...       [[14, 17],
    ...        [15, 16]]]])
    >>> res = chi2_contingency(obs)
    >>> res.statistic
    8.7584514426741897
    >>> res.pvalue
    0.64417725029295503
    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError('All values in `observed` must be nonnegative.')
    if observed.size == 0:
        raise ValueError('No data; `observed` has size 0.')
    expected = expected_freq(observed)
    if np.any(expected == 0):
        zeropos = list(zip(*np.nonzero(expected == 0)))[0]
        raise ValueError(f'The internally computed table of expected frequencies has a zero element at {zeropos}.')
    dof = expected.size - sum(expected.shape) + expected.ndim - 1
    if dof == 0:
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            diff = expected - observed
            direction = np.sign(diff)
            magnitude = np.minimum(0.5, np.abs(diff))
            observed = observed + magnitude * direction
        chi2, p = power_divergence(observed, expected, ddof=observed.size - 1 - dof, axis=None, lambda_=lambda_)
    return Chi2ContingencyResult(chi2, p, dof, expected)