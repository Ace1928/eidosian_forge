from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def homogeneous_subsets(vals, dcrit):
    """recursively check all pairs of vals for minimum distance

    step down method as in Newman-Keuls and Ryan procedures. This is not a
    closed procedure since not all partitions are checked.

    Parameters
    ----------
    vals : array_like
        values that are pairwise compared
    dcrit : array_like or float
        critical distance for rejecting, either float, or 2-dimensional array
        with distances on the upper triangle.

    Returns
    -------
    rejs : list of pairs
        list of pair-indices with (strictly) larger than critical difference
    nrejs : list of pairs
        list of pair-indices with smaller than critical difference
    lli : list of tuples
        list of subsets with smaller than critical difference
    res : tree
        result of all comparisons (for checking)


    this follows description in SPSS notes on Post-Hoc Tests

    Because of the recursive structure, some comparisons are made several
    times, but only unique pairs or sets are returned.

    Examples
    --------
    >>> m = [0, 2, 2.5, 3, 6, 8, 9, 9.5,10 ]
    >>> rej, nrej, ssli, res = homogeneous_subsets(m, 2)
    >>> set_partition(ssli)
    ([(5, 6, 7, 8), (1, 2, 3), (4,)], [0])
    >>> [np.array(m)[list(pp)] for pp in set_partition(ssli)[0]]
    [array([  8. ,   9. ,   9.5,  10. ]), array([ 2. ,  2.5,  3. ]), array([ 6.])]


    """
    nvals = len(vals)
    indices_ = lrange(nvals)
    rejected = []
    subsetsli = []
    if np.size(dcrit) == 1:
        dcrit = dcrit * np.ones((nvals, nvals))

    def subsets(vals, indices_):
        """recursive function for constructing homogeneous subset

        registers rejected and subsetli in outer scope
        """
        i, j = (indices_[0], indices_[-1])
        if vals[-1] - vals[0] > dcrit[i, j]:
            rejected.append((indices_[0], indices_[-1]))
            return [subsets(vals[:-1], indices_[:-1]), subsets(vals[1:], indices_[1:]), (indices_[0], indices_[-1])]
        else:
            subsetsli.append(tuple(indices_))
            return indices_
    res = subsets(vals, indices_)
    all_pairs = [(i, j) for i in range(nvals) for j in range(nvals - 1, i, -1)]
    rejs = set(rejected)
    not_rejected = list(set(all_pairs) - rejs)
    return (list(rejs), not_rejected, list(set(subsetsli)), res)