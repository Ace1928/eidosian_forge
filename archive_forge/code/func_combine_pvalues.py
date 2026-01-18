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
def combine_pvalues(pvalues, method='fisher', weights=None):
    """
    Combine p-values from independent tests that bear upon the same hypothesis.

    These methods are intended only for combining p-values from hypothesis
    tests based upon continuous distributions.

    Each method assumes that under the null hypothesis, the p-values are
    sampled independently and uniformly from the interval [0, 1]. A test
    statistic (different for each method) is computed and a combined
    p-value is calculated based upon the distribution of this test statistic
    under the null hypothesis.

    Parameters
    ----------
    pvalues : array_like, 1-D
        Array of p-values assumed to come from independent tests based on
        continuous distributions.
    method : {'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george'}

        Name of method to use to combine p-values.

        The available methods are (see Notes for details):

        * 'fisher': Fisher's method (Fisher's combined probability test)
        * 'pearson': Pearson's method
        * 'mudholkar_george': Mudholkar's and George's method
        * 'tippett': Tippett's method
        * 'stouffer': Stouffer's Z-score method
    weights : array_like, 1-D, optional
        Optional array of weights used only for Stouffer's Z-score method.

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
            The statistic calculated by the specified method.
        pvalue : float
            The combined p-value.

    Examples
    --------
    Suppose we wish to combine p-values from four independent tests
    of the same null hypothesis using Fisher's method (default).

    >>> from scipy.stats import combine_pvalues
    >>> pvalues = [0.1, 0.05, 0.02, 0.3]
    >>> combine_pvalues(pvalues)
    SignificanceResult(statistic=20.828626352604235, pvalue=0.007616871850449092)

    When the individual p-values carry different weights, consider Stouffer's
    method.

    >>> weights = [1, 2, 3, 4]
    >>> res = combine_pvalues(pvalues, method='stouffer', weights=weights)
    >>> res.pvalue
    0.009578891494533616

    Notes
    -----
    If this function is applied to tests with a discrete statistics such as
    any rank test or contingency-table test, it will yield systematically
    wrong results, e.g. Fisher's method will systematically overestimate the
    p-value [1]_. This problem becomes less severe for large sample sizes
    when the discrete distributions become approximately continuous.

    The differences between the methods can be best illustrated by their
    statistics and what aspects of a combination of p-values they emphasise
    when considering significance [2]_. For example, methods emphasising large
    p-values are more sensitive to strong false and true negatives; conversely
    methods focussing on small p-values are sensitive to positives.

    * The statistics of Fisher's method (also known as Fisher's combined
      probability test) [3]_ is :math:`-2\\sum_i \\log(p_i)`, which is
      equivalent (as a test statistics) to the product of individual p-values:
      :math:`\\prod_i p_i`. Under the null hypothesis, this statistics follows
      a :math:`\\chi^2` distribution. This method emphasises small p-values.
    * Pearson's method uses :math:`-2\\sum_i\\log(1-p_i)`, which is equivalent
      to :math:`\\prod_i \\frac{1}{1-p_i}` [2]_.
      It thus emphasises large p-values.
    * Mudholkar and George compromise between Fisher's and Pearson's method by
      averaging their statistics [4]_. Their method emphasises extreme
      p-values, both close to 1 and 0.
    * Stouffer's method [5]_ uses Z-scores and the statistic:
      :math:`\\sum_i \\Phi^{-1} (p_i)`, where :math:`\\Phi` is the CDF of the
      standard normal distribution. The advantage of this method is that it is
      straightforward to introduce weights, which can make Stouffer's method
      more powerful than Fisher's method when the p-values are from studies
      of different size [6]_ [7]_.
    * Tippett's method uses the smallest p-value as a statistic.
      (Mind that this minimum is not the combined p-value.)

    Fisher's method may be extended to combine p-values from dependent tests
    [8]_. Extensions such as Brown's method and Kost's method are not currently
    implemented.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Kincaid, W. M., "The Combination of Tests Based on Discrete
           Distributions." Journal of the American Statistical Association 57,
           no. 297 (1962), 10-19.
    .. [2] Heard, N. and Rubin-Delanchey, P. "Choosing between methods of
           combining p-values."  Biometrika 105.1 (2018): 239-246.
    .. [3] https://en.wikipedia.org/wiki/Fisher%27s_method
    .. [4] George, E. O., and G. S. Mudholkar. "On the convolution of logistic
           random variables." Metrika 30.1 (1983): 1-13.
    .. [5] https://en.wikipedia.org/wiki/Fisher%27s_method#Relation_to_Stouffer.27s_Z-score_method
    .. [6] Whitlock, M. C. "Combining probability from independent tests: the
           weighted Z-method is superior to Fisher's approach." Journal of
           Evolutionary Biology 18, no. 5 (2005): 1368-1373.
    .. [7] Zaykin, Dmitri V. "Optimally weighted Z-test is a powerful method
           for combining probabilities in meta-analysis." Journal of
           Evolutionary Biology 24, no. 8 (2011): 1836-1841.
    .. [8] https://en.wikipedia.org/wiki/Extensions_of_Fisher%27s_method

    """
    pvalues = np.asarray(pvalues)
    if pvalues.ndim != 1:
        raise ValueError('pvalues is not 1-D')
    if method == 'fisher':
        statistic = -2 * np.sum(np.log(pvalues))
        pval = distributions.chi2.sf(statistic, 2 * len(pvalues))
    elif method == 'pearson':
        statistic = 2 * np.sum(np.log1p(-pvalues))
        pval = distributions.chi2.cdf(-statistic, 2 * len(pvalues))
    elif method == 'mudholkar_george':
        normalizing_factor = np.sqrt(3 / len(pvalues)) / np.pi
        statistic = -np.sum(np.log(pvalues)) + np.sum(np.log1p(-pvalues))
        nu = 5 * len(pvalues) + 4
        approx_factor = np.sqrt(nu / (nu - 2))
        pval = distributions.t.sf(statistic * normalizing_factor * approx_factor, nu)
    elif method == 'tippett':
        statistic = np.min(pvalues)
        pval = distributions.beta.cdf(statistic, 1, len(pvalues))
    elif method == 'stouffer':
        if weights is None:
            weights = np.ones_like(pvalues)
        elif len(weights) != len(pvalues):
            raise ValueError('pvalues and weights must be of the same size.')
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError('weights is not 1-D')
        Zi = distributions.norm.isf(pvalues)
        statistic = np.dot(weights, Zi) / np.linalg.norm(weights)
        pval = distributions.norm.sf(statistic)
    else:
        raise ValueError(f"Invalid method {method!r}. Valid methods are 'fisher', 'pearson', 'mudholkar_george', 'tippett', and 'stouffer'")
    return SignificanceResult(statistic, pval)