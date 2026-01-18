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
def mood(x, y, axis=0, alternative='two-sided'):
    """Perform Mood's test for equal scale parameters.

    Mood's two-sample test for scale parameters is a non-parametric
    test for the null hypothesis that two samples are drawn from the
    same distribution with the same scale parameter.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data.
    axis : int, optional
        The axis along which the samples are tested.  `x` and `y` can be of
        different length along `axis`.
        If `axis` is None, `x` and `y` are flattened and the test is done on
        all values in the flattened arrays.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the scales of the distributions underlying `x` and `y`
          are different.
        * 'less': the scale of the distribution underlying `x` is less than
          the scale of the distribution underlying `y`.
        * 'greater': the scale of the distribution underlying `x` is greater
          than the scale of the distribution underlying `y`.

        .. versionadded:: 1.7.0

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : scalar or ndarray
            The z-score for the hypothesis test.  For 1-D inputs a scalar is
            returned.
        pvalue : scalar ndarray
            The p-value for the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    ansari : A non-parametric test for the equality of 2 variances
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A parametric test for equality of k variances

    Notes
    -----
    The data are assumed to be drawn from probability distributions ``f(x)``
    and ``f(x/s) / s`` respectively, for some probability density function f.
    The null hypothesis is that ``s == 1``.

    For multi-dimensional arrays, if the inputs are of shapes
    ``(n0, n1, n2, n3)``  and ``(n0, m1, n2, n3)``, then if ``axis=1``, the
    resulting z and p values will have shape ``(n0, n2, n3)``.  Note that
    ``n1`` and ``m1`` don't have to be equal, but the other dimensions do.

    References
    ----------
    [1] Mielke, Paul W. "Note on Some Squared Rank Tests with Existing Ties."
        Technometrics, vol. 9, no. 2, 1967, pp. 312-14. JSTOR,
        https://doi.org/10.2307/1266427. Accessed 18 May 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x2 = rng.standard_normal((2, 45, 6, 7))
    >>> x1 = rng.standard_normal((2, 30, 6, 7))
    >>> res = stats.mood(x1, x2, axis=1)
    >>> res.pvalue.shape
    (2, 6, 7)

    Find the number of points where the difference in scale is not significant:

    >>> (res.pvalue > 0.1).sum()
    78

    Perform the test with different scales:

    >>> x1 = rng.standard_normal((2, 30))
    >>> x2 = rng.standard_normal((2, 35)) * 10.0
    >>> stats.mood(x1, x2, axis=1)
    SignificanceResult(statistic=array([-5.76174136, -6.12650783]),
                       pvalue=array([8.32505043e-09, 8.98287869e-10]))

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if axis is None:
        x = x.flatten()
        y = y.flatten()
        axis = 0
    if axis < 0:
        axis = x.ndim + axis
    res_shape = tuple([x.shape[ax] for ax in range(len(x.shape)) if ax != axis])
    if not res_shape == tuple([y.shape[ax] for ax in range(len(y.shape)) if ax != axis]):
        raise ValueError('Dimensions of x and y on all axes except `axis` should match')
    n = x.shape[axis]
    m = y.shape[axis]
    N = m + n
    if N < 3:
        raise ValueError('Not enough observations.')
    xy = np.concatenate((x, y), axis=axis)
    sorted_xy = np.sort(xy, axis=axis)
    diffs = np.diff(sorted_xy, axis=axis)
    if 0 in diffs:
        z = np.asarray(_mood_inner_lc(xy, x, diffs, sorted_xy, n, m, N, axis=axis))
    else:
        if axis != 0:
            xy = np.moveaxis(xy, axis, 0)
        xy = xy.reshape(xy.shape[0], -1)
        all_ranks = np.empty_like(xy)
        for j in range(xy.shape[1]):
            all_ranks[:, j] = _stats_py.rankdata(xy[:, j])
        Ri = all_ranks[:n]
        M = np.sum((Ri - (N + 1.0) / 2) ** 2, axis=0)
        mnM = n * (N * N - 1.0) / 12
        varM = m * n * (N + 1.0) * (N + 2) * (N - 2) / 180
        z = (M - mnM) / sqrt(varM)
    z, pval = _normtest_finish(z, alternative)
    if res_shape == ():
        z = z[0]
        pval = pval[0]
    else:
        z.shape = res_shape
        pval.shape = res_shape
    return SignificanceResult(z, pval)