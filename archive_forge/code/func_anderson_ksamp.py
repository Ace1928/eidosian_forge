from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def anderson_ksamp(samples, midrank=True, *, method=None):
    """The Anderson-Darling test for k-samples.

    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample data in arrays.
    midrank : bool, optional
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.
    method : PermutationMethod, optional
        Defines the method used to compute the p-value. If `method` is an
        instance of `PermutationMethod`, the p-value is computed using
        `scipy.stats.permutation_test` with the provided configuration options
        and other appropriate settings. Otherwise, the p-value is interpolated
        from tabulated values.

    Returns
    -------
    res : Anderson_ksampResult
        An object containing attributes:

        statistic : float
            Normalized k-sample Anderson-Darling test statistic.
        critical_values : array
            The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
            0.5%, 0.1%.
        pvalue : float
            The approximate p-value of the test. If `method` is not
            provided, the value is floored / capped at 0.1% / 25%.

    Raises
    ------
    ValueError
        If fewer than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.

    See Also
    --------
    ks_2samp : 2 sample Kolmogorov-Smirnov test
    anderson : 1 sample Anderson-Darling test

    Notes
    -----
    [1]_ defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.

    The critical values corresponding to the significance levels from 0.01
    to 0.25 are taken from [1]_. p-values are floored / capped
    at 0.1% / 25%. Since the range of critical values might be extended in
    future releases, it is recommended not to test ``p == 0.25``, but rather
    ``p >= 0.25`` (analogously for the lower bound).

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> res = stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(loc=0.5, size=30)])
    >>> res.statistic, res.pvalue
    (1.974403288713695, 0.04991293614572478)
    >>> res.critical_values
    array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546])

    The null hypothesis that the two random samples come from the same
    distribution can be rejected at the 5% level because the returned
    test value is greater than the critical value for 5% (1.961) but
    not at the 2.5% level. The interpolation gives an approximate
    p-value of 4.99%.

    >>> samples = [rng.normal(size=50), rng.normal(size=30),
    ...            rng.normal(size=20)]
    >>> res = stats.anderson_ksamp(samples)
    >>> res.statistic, res.pvalue
    (-0.29103725200789504, 0.25)
    >>> res.critical_values
    array([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856,
      4.07210043, 5.56419101])

    The null hypothesis cannot be rejected for three samples from an
    identical distribution. The reported p-value (25%) has been capped and
    may not be very accurate (since it corresponds to the value 0.449
    whereas the statistic is -0.291).

    In such cases where the p-value is capped or when sample sizes are
    small, a permutation test may be more accurate.

    >>> method = stats.PermutationMethod(n_resamples=9999, random_state=rng)
    >>> res = stats.anderson_ksamp(samples, method=method)
    >>> res.pvalue
    0.5254

    """
    k = len(samples)
    if k < 2:
        raise ValueError('anderson_ksamp needs at least two samples')
    samples = list(map(np.asarray, samples))
    Z = np.sort(np.hstack(samples))
    N = Z.size
    Zstar = np.unique(Z)
    if Zstar.size < 2:
        raise ValueError('anderson_ksamp needs more than one distinct observation')
    n = np.array([sample.size for sample in samples])
    if np.any(n == 0):
        raise ValueError('anderson_ksamp encountered sample without observations')
    if midrank:
        A2kN_fun = _anderson_ksamp_midrank
    else:
        A2kN_fun = _anderson_ksamp_right
    A2kN = A2kN_fun(samples, Z, Zstar, k, n, N)

    def statistic(*samples):
        return A2kN_fun(samples, Z, Zstar, k, n, N)
    if method is not None:
        res = stats.permutation_test(samples, statistic, **method._asdict(), alternative='greater')
    H = (1.0 / n).sum()
    hs_cs = (1.0 / arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / arange(2, N)).sum()
    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * k ** 2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k ** 2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * k ** 2 - 4 * h * k
    sigmasq = (a * N ** 3 + b * N ** 2 + c * N + d) / ((N - 1.0) * (N - 2.0) * (N - 3.0))
    m = k - 1
    A2 = (A2kN - m) / math.sqrt(sigmasq)
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    critical = b0 + b1 / math.sqrt(m) + b2 / m
    sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    if A2 < critical.min() and method is None:
        p = sig.max()
        msg = f'p-value capped: true value larger than {p}. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)'
        warnings.warn(msg, stacklevel=2)
    elif A2 > critical.max() and method is None:
        p = sig.min()
        msg = f'p-value floored: true value smaller than {p}. Consider specifying `method` (e.g. `method=stats.PermutationMethod()`.)'
        warnings.warn(msg, stacklevel=2)
    elif method is None:
        pf = np.polyfit(critical, log(sig), 2)
        p = math.exp(np.polyval(pf, A2))
    else:
        p = res.pvalue if method is not None else p
    res = Anderson_ksampResult(A2, critical, p)
    res.significance_level = p
    return res