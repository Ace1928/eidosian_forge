import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def kruskal(*args):
    """
    Compute the Kruskal-Wallis H-test for independent samples

    Parameters
    ----------
    sample1, sample2, ... : array_like
       Two or more arrays with the sample measurements can be given as
       arguments.

    Returns
    -------
    statistic : float
       The Kruskal-Wallis H statistic, corrected for ties
    pvalue : float
       The p-value for the test using the assumption that H has a chi
       square distribution

    Notes
    -----
    For more details on `kruskal`, see `scipy.stats.kruskal`.

    Examples
    --------
    >>> from scipy.stats.mstats import kruskal

    Random samples from three different brands of batteries were tested
    to see how long the charge lasted. Results were as follows:

    >>> a = [6.3, 5.4, 5.7, 5.2, 5.0]
    >>> b = [6.9, 7.0, 6.1, 7.9]
    >>> c = [7.2, 6.9, 6.1, 6.5]

    Test the hypothesis that the distribution functions for all of the brands'
    durations are identical. Use 5% level of significance.

    >>> kruskal(a, b, c)
    KruskalResult(statistic=7.113812154696133, pvalue=0.028526948491942164)

    The null hypothesis is rejected at the 5% level of significance
    because the returned p-value is less than the critical value of 5%.

    """
    output = argstoarray(*args)
    ranks = ma.masked_equal(rankdata(output, use_missing=False), 0)
    sumrk = ranks.sum(-1)
    ngrp = ranks.count(-1)
    ntot = ranks.count()
    H = 12.0 / (ntot * (ntot + 1)) * (sumrk ** 2 / ngrp).sum() - 3 * (ntot + 1)
    ties = count_tied_groups(ranks)
    T = 1.0 - sum((v * (k ** 3 - k) for k, v in ties.items())) / float(ntot ** 3 - ntot)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')
    H /= T
    df = len(output) - 1
    prob = distributions.chi2.sf(H, df)
    return KruskalResult(H, prob)