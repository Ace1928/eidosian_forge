from functools import reduce
import math
import numpy as np
from ._stats_py import power_divergence
from ._relative_risk import relative_risk
from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from scipy._lib._bunch import _make_tuple_bunch
def expected_freq(observed):
    """
    Compute the expected frequencies from a contingency table.

    Given an n-dimensional contingency table of observed frequencies,
    compute the expected frequencies for the table based on the marginal
    sums under the assumption that the groups associated with each
    dimension are independent.

    Parameters
    ----------
    observed : array_like
        The table of observed frequencies.  (While this function can handle
        a 1-D array, that case is trivial.  Generally `observed` is at
        least 2-D.)

    Returns
    -------
    expected : ndarray of float64
        The expected frequencies, based on the marginal sums of the table.
        Same shape as `observed`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import expected_freq
    >>> observed = np.array([[10, 10, 20],[20, 20, 20]])
    >>> expected_freq(observed)
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    """
    observed = np.asarray(observed, dtype=np.float64)
    margsums = margins(observed)
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected