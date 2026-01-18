from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def proportions_chisquare_pairscontrol(count, nobs, value=None, multitest_method='hs', alternative='two-sided'):
    """
    Chisquare test of proportions for pairs of k samples compared to control

    Performs a chisquare test for proportions for pairwise comparisons with a
    control (Dunnet's test). The control is assumed to be the first element
    of ``count`` and ``nobs``. The alternative is two-sided, larger or
    smaller.

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials.
    nobs : int
        the number of trials or observations.
    multitest_method : str
        This chooses the method for the multiple testing p-value correction,
        that is used as default in the results.
        It can be any method that is available in  ``multipletesting``.
        The default is Holm-Sidak 'hs'.
    alternative : str in ['two-sided', 'smaller', 'larger']
        alternative hypothesis, which can be two-sided or either one of the
        one-sided tests.

    Returns
    -------
    result : AllPairsResults instance
        The returned results instance has several statistics, such as p-values,
        attached, and additional methods for using a non-default
        ``multitest_method``.


    Notes
    -----
    Yates continuity correction is not available.

    ``value`` and ``alternative`` options are not yet implemented.

    """
    if value is not None or alternative not in ['two-sided', '2s']:
        raise NotImplementedError
    all_pairs = [(0, k) for k in range(1, len(count))]
    pvals = [proportions_chisquare(count[list(pair)], nobs[list(pair)])[1] for pair in all_pairs]
    return AllPairsResults(pvals, all_pairs, multitest_method=multitest_method)