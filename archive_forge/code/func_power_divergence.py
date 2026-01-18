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
def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    """Cressie-Read power divergence statistic and goodness of fit test.

    This function tests the null hypothesis that the categorical data
    has the given frequencies, using the Cressie-Read power divergence
    statistic.

    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.
    lambda_ : float or str, optional
        The power in the Cressie-Read power divergence statistic.  The default
        is 1.  For convenience, `lambda_` may be assigned one of the following
        strings, in which case the corresponding numerical value is used:

        * ``"pearson"`` (value 1)
            Pearson's chi-squared statistic. In this case, the function is
            equivalent to `chisquare`.
        * ``"log-likelihood"`` (value 0)
            Log-likelihood ratio. Also known as the G-test [3]_.
        * ``"freeman-tukey"`` (value -1/2)
            Freeman-Tukey statistic.
        * ``"mod-log-likelihood"`` (value -1)
            Modified log-likelihood ratio.
        * ``"neyman"`` (value -2)
            Neyman's statistic.
        * ``"cressie-read"`` (value 2/3)
            The power recommended in [5]_.

    Returns
    -------
    res: Power_divergenceResult
        An object containing attributes:

        statistic : float or ndarray
            The Cressie-Read power divergence test statistic.  The value is
            a float if `axis` is None or if` `f_obs` and `f_exp` are 1-D.
        pvalue : float or ndarray
            The p-value of the test.  The value is a float if `ddof` and the
            return value `stat` are scalars.

    See Also
    --------
    chisquare

    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5.

    Also, the sum of the observed and expected frequencies must be the same
    for the test to be valid; `power_divergence` raises an error if the sums
    do not agree within a relative tolerance of ``1e-8``.

    When `lambda_` is less than zero, the formula for the statistic involves
    dividing by `f_obs`, so a warning or error may be generated if any value
    in `f_obs` is 0.

    Similarly, a warning or error may be generated if any value in `f_exp` is
    zero when `lambda_` >= 0.

    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not a chisquare, in which case this
    test is not appropriate.

    This function handles masked arrays.  If an element of `f_obs` or `f_exp`
    is masked, then data at that position is ignored, and does not count
    towards the size of the data set.

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171015035606/http://faculty.vassar.edu/lowry/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] "G-test", https://en.wikipedia.org/wiki/G-test
    .. [4] Sokal, R. R. and Rohlf, F. J. "Biometry: the principles and
           practice of statistics in biological research", New York: Freeman
           (1981)
    .. [5] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.

    Examples
    --------
    (See `chisquare` for more examples.)

    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.  Here we
    perform a G-test (i.e. use the log-likelihood ratio statistic):

    >>> import numpy as np
    >>> from scipy.stats import power_divergence
    >>> power_divergence([16, 18, 16, 14, 12, 12], lambda_='log-likelihood')
    (2.006573162632538, 0.84823476779463769)

    The expected frequencies can be given with the `f_exp` argument:

    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[16, 16, 16, 16, 16, 8],
    ...                  lambda_='log-likelihood')
    (3.3281031458963746, 0.6495419288047497)

    When `f_obs` is 2-D, by default the test is applied to each column.

    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> power_divergence(obs, lambda_="log-likelihood")
    (array([ 2.00657316,  6.77634498]), array([ 0.84823477,  0.23781225]))

    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.

    >>> power_divergence(obs, axis=None)
    (23.31034482758621, 0.015975692534127565)
    >>> power_divergence(obs.ravel())
    (23.31034482758621, 0.015975692534127565)

    `ddof` is the change to make to the default degrees of freedom.

    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=1)
    (2.0, 0.73575888234288467)

    The calculation of the p-values is done by broadcasting the
    test statistic with `ddof`.

    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=[0,1,2])
    (2.0, array([ 0.84914504,  0.73575888,  0.5724067 ]))

    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we must use ``axis=1``:

    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[[16, 16, 16, 16, 16, 8],
    ...                         [8, 20, 20, 16, 12, 12]],
    ...                  axis=1)
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))

    """
    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError(f'invalid string for lambda_: {lambda_!r}. Valid strings are {names}')
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1
    f_obs = np.asanyarray(f_obs)
    f_obs_float = f_obs.astype(np.float64)
    if f_exp is not None:
        f_exp = np.asanyarray(f_exp)
        bshape = np.broadcast_shapes(f_obs_float.shape, f_exp.shape)
        f_obs_float = _m_broadcast_to(f_obs_float, bshape)
        f_exp = _m_broadcast_to(f_exp, bshape)
        rtol = 1e-08
        with np.errstate(invalid='ignore'):
            f_obs_sum = f_obs_float.sum(axis=axis)
            f_exp_sum = f_exp.sum(axis=axis)
            relative_diff = np.abs(f_obs_sum - f_exp_sum) / np.minimum(f_obs_sum, f_exp_sum)
            diff_gt_tol = (relative_diff > rtol).any()
        if diff_gt_tol:
            msg = f'For each axis slice, the sum of the observed frequencies must agree with the sum of the expected frequencies to a relative tolerance of {rtol}, but the percent differences are:\n{relative_diff}'
            raise ValueError(msg)
    else:
        with np.errstate(invalid='ignore'):
            f_exp = f_obs.mean(axis=axis, keepdims=True)
    if lambda_ == 1:
        terms = (f_obs_float - f_exp) ** 2 / f_exp
    elif lambda_ == 0:
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        terms = f_obs * ((f_obs / f_exp) ** lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)
    stat = terms.sum(axis=axis)
    num_obs = _count(terms, axis=axis)
    ddof = asarray(ddof)
    p = distributions.chi2.sf(stat, num_obs - 1 - ddof)
    return Power_divergenceResult(stat, p)