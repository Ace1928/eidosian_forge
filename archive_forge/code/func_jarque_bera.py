import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
@_axis_nan_policy_factory(SignificanceResult, default_axis=None)
def jarque_bera(x, *, axis=None):
    """Perform the Jarque-Bera goodness of fit test on sample data.

    The Jarque-Bera test tests whether the sample data has the skewness and
    kurtosis matching a normal distribution.

    Note that this test only works for a large enough number of data samples
    (>2000) as the test statistic asymptotically has a Chi-squared distribution
    with 2 degrees of freedom.

    Parameters
    ----------
    x : array_like
        Observations of a random variable.
    axis : int or None, default: 0
        If an int, the axis of the input along which to compute the statistic.
        The statistic of each axis-slice (e.g. row) of the input will appear in
        a corresponding element of the output.
        If ``None``, the input will be raveled before computing the statistic.

    Returns
    -------
    result : SignificanceResult
        An object with the following attributes:

        statistic : float
            The test statistic.
        pvalue : float
            The p-value for the hypothesis test.

    References
    ----------
    .. [1] Jarque, C. and Bera, A. (1980) "Efficient tests for normality,
           homoscedasticity and serial independence of regression residuals",
           6 Econometric Letters 255-259.
    .. [2] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591-611.
    .. [3] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [4] Panagiotakos, D. B. (2008). The value of p-value in biomedical
           research. The open cardiovascular medicine journal, 2, 97.

    Examples
    --------
    Suppose we wish to infer from measurements whether the weights of adult
    human males in a medical study are not normally distributed [2]_.
    The weights (lbs) are recorded in the array ``x`` below.

    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])

    The Jarque-Bera test begins by computing a statistic based on the sample
    skewness and kurtosis.

    >>> from scipy import stats
    >>> res = stats.jarque_bera(x)
    >>> res.statistic
    6.982848237344646

    Because the normal distribution has zero skewness and zero
    ("excess" or "Fisher") kurtosis, the value of this statistic tends to be
    low for samples drawn from a normal distribution.

    The test is performed by comparing the observed value of the statistic
    against the null distribution: the distribution of statistic values derived
    under the null hypothesis that the weights were drawn from a normal
    distribution.
    For the Jarque-Bera test, the null distribution for very large samples is
    the chi-squared distribution with two degrees of freedom.

    >>> import matplotlib.pyplot as plt
    >>> dist = stats.chi2(df=2)
    >>> jb_val = np.linspace(0, 11, 100)
    >>> pdf = dist.pdf(jb_val)
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> def jb_plot(ax):  # we'll reuse this
    ...     ax.plot(jb_val, pdf)
    ...     ax.set_title("Jarque-Bera Null Distribution")
    ...     ax.set_xlabel("statistic")
    ...     ax.set_ylabel("probability density")
    >>> jb_plot(ax)
    >>> plt.show()

    The comparison is quantified by the p-value: the proportion of values in
    the null distribution greater than or equal to the observed value of the
    statistic.

    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> jb_plot(ax)
    >>> pvalue = dist.sf(res.statistic)
    >>> annotation = (f'p-value={pvalue:.6f}\\n(shaded area)')
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    >>> _ = ax.annotate(annotation, (7.5, 0.01), (8, 0.05), arrowprops=props)
    >>> i = jb_val >= res.statistic  # indices of more extreme statistic values
    >>> ax.fill_between(jb_val[i], y1=0, y2=pdf[i])
    >>> ax.set_xlim(0, 11)
    >>> ax.set_ylim(0, 0.3)
    >>> plt.show()
    >>> res.pvalue
    0.03045746622458189

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from a normally distributed population that produces such an
    extreme value of the statistic - this may be taken as evidence against
    the null hypothesis in favor of the alternative: the weights were not
    drawn from a normal distribution. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence for the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [3]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    Note that the chi-squared distribution provides an asymptotic approximation
    of the null distribution; it is only accurate for samples with many
    observations. For small samples like ours, `scipy.stats.monte_carlo_test`
    may provide a more accurate, albeit stochastic, approximation of the
    exact p-value.

    >>> def statistic(x, axis):
    ...     # underlying calculation of the Jarque Bera statistic
    ...     s = stats.skew(x, axis=axis)
    ...     k = stats.kurtosis(x, axis=axis)
    ...     return x.shape[axis]/6 * (s**2 + k**2/4)
    >>> res = stats.monte_carlo_test(x, stats.norm.rvs, statistic,
    ...                              alternative='greater')
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> jb_plot(ax)
    >>> ax.hist(res.null_distribution, np.linspace(0, 10, 50),
    ...         density=True)
    >>> ax.legend(['aymptotic approximation (many observations)',
    ...            'Monte Carlo approximation (11 observations)'])
    >>> plt.show()
    >>> res.pvalue
    0.0097  # may vary

    Furthermore, despite their stochastic nature, p-values computed in this way
    can be used to exactly control the rate of false rejections of the null
    hypothesis [4]_.

    """
    x = np.asarray(x)
    if axis is None:
        x = x.ravel()
        axis = 0
    n = x.shape[axis]
    if n == 0:
        raise ValueError('At least one observation is required.')
    mu = x.mean(axis=axis, keepdims=True)
    diffx = x - mu
    s = skew(diffx, axis=axis, _no_deco=True)
    k = kurtosis(diffx, axis=axis, _no_deco=True)
    statistic = n / 6 * (s ** 2 + k ** 2 / 4)
    pvalue = distributions.chi2.sf(statistic, df=2)
    return SignificanceResult(statistic, pvalue)