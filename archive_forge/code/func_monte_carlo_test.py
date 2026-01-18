from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
@_rename_parameter('sample', 'data')
def monte_carlo_test(data, rvs, statistic, *, vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=0):
    """Perform a Monte Carlo hypothesis test.

    `data` contains a sample or a sequence of one or more samples. `rvs`
    specifies the distribution(s) of the sample(s) in `data` under the null
    hypothesis. The value of `statistic` for the given `data` is compared
    against a Monte Carlo null distribution: the value of the statistic for
    each of `n_resamples` sets of samples generated using `rvs`. This gives
    the p-value, the probability of observing such an extreme value of the
    test statistic under the null hypothesis.

    Parameters
    ----------
    data : array-like or sequence of array-like
        An array or sequence of arrays of observations.
    rvs : callable or tuple of callables
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples in
        `data`, i.e. ``len(rvs) == len(data)``. If `rvs` is a single callable,
        `data` is treated as a single sample.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts a sample
        (e.g. ``statistic(sample)``) or ``len(rvs)`` separate samples (e.g.
        ``statistic(samples1, sample2)`` if `rvs` contains two callables and
        `data` contains two samples) and returns the resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the samples in `data`.
    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed ND sample arrays. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of
        a vectorized statistic typically reduces computation time.
    n_resamples : int, default: 9999
        Number of samples drawn from each of the callables of `rvs`.
        Equivalently, the number statistic values under the null hypothesis
        used as the Monte Carlo null distribution.
    batch : int, optional
        The number of Monte Carlo samples to process in each call to
        `statistic`. Memory usage is O( `batch` * ``sample.size[axis]`` ). Default
        is ``None``, in which case `batch` equals `n_resamples`.
    alternative : {'two-sided', 'less', 'greater'}
        The alternative hypothesis for which the p-value is calculated.
        For each alternative, the p-value is defined as follows.

        - ``'greater'`` : the percentage of the null distribution that is
          greater than or equal to the observed value of the test statistic.
        - ``'less'`` : the percentage of the null distribution that is
          less than or equal to the observed value of the test statistic.
        - ``'two-sided'`` : twice the smaller of the p-values above.

    axis : int, default: 0
        The axis of `data` (or each sample within `data`) over which to
        calculate the statistic.

    Returns
    -------
    res : MonteCarloTestResult
        An object with attributes:

        statistic : float or ndarray
            The test statistic of the observed `data`.
        pvalue : float or ndarray
            The p-value for the given alternative.
        null_distribution : ndarray
            The values of the test statistic generated under the null
            hypothesis.

    References
    ----------

    .. [1] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).

    Examples
    --------

    Suppose we wish to test whether a small sample has been drawn from a normal
    distribution. We decide that we will use the skew of the sample as a
    test statistic, and we will consider a p-value of 0.05 to be statistically
    significant.

    >>> import numpy as np
    >>> from scipy import stats
    >>> def statistic(x, axis):
    ...     return stats.skew(x, axis)

    After collecting our data, we calculate the observed value of the test
    statistic.

    >>> rng = np.random.default_rng()
    >>> x = stats.skewnorm.rvs(a=1, size=50, random_state=rng)
    >>> statistic(x, axis=0)
    0.12457412450240658

    To determine the probability of observing such an extreme value of the
    skewness by chance if the sample were drawn from the normal distribution,
    we can perform a Monte Carlo hypothesis test. The test will draw many
    samples at random from their normal distribution, calculate the skewness
    of each sample, and compare our original skewness against this
    distribution to determine an approximate p-value.

    >>> from scipy.stats import monte_carlo_test
    >>> # because our statistic is vectorized, we pass `vectorized=True`
    >>> rvs = lambda size: stats.norm.rvs(size=size, random_state=rng)
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)
    >>> print(res.statistic)
    0.12457412450240658
    >>> print(res.pvalue)
    0.7012

    The probability of obtaining a test statistic less than or equal to the
    observed value under the null hypothesis is ~70%. This is greater than
    our chosen threshold of 5%, so we cannot consider this to be significant
    evidence against the null hypothesis.

    Note that this p-value essentially matches that of
    `scipy.stats.skewtest`, which relies on an asymptotic distribution of a
    test statistic based on the sample skewness.

    >>> stats.skewtest(x).pvalue
    0.6892046027110614

    This asymptotic approximation is not valid for small sample sizes, but
    `monte_carlo_test` can be used with samples of any size.

    >>> x = stats.skewnorm.rvs(a=1, size=7, random_state=rng)
    >>> # stats.skewtest(x) would produce an error due to small sample
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)

    The Monte Carlo distribution of the test statistic is provided for
    further investigation.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.null_distribution, bins=50)
    >>> ax.set_title("Monte Carlo distribution of test statistic")
    >>> ax.set_xlabel("Value of Statistic")
    >>> ax.set_ylabel("Frequency")
    >>> plt.show()

    """
    args = _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples, batch, alternative, axis)
    data, rvs, statistic, vectorized, n_resamples, batch, alternative, axis = args
    observed = np.asarray(statistic(*data, axis=-1))[()]
    n_observations = [sample.shape[-1] for sample in data]
    batch_nominal = batch or n_resamples
    null_distribution = []
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        resamples = [rvs_i(size=(batch_actual, n_observations_i)) for rvs_i, n_observations_i in zip(rvs, n_observations)]
        null_distribution.append(statistic(*resamples, axis=-1))
    null_distribution = np.concatenate(null_distribution)
    null_distribution = null_distribution.reshape([-1] + [1] * observed.ndim)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed
        pvalues = (cmps.sum(axis=0) + 1) / (n_resamples + 1)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed
        pvalues = (cmps.sum(axis=0) + 1) / (n_resamples + 1)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)
    return MonteCarloTestResult(observed, pvalues, null_distribution)