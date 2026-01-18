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
def set_remove_subs(ssli):
    """remove sets that are subsets of another set from a list of tuples

    Parameters
    ----------
    ssli : list of tuples
        each tuple is considered as a set

    Returns
    -------
    part : list of tuples
        new list with subset tuples removed, it is sorted by set-length of tuples. The
        list contains original tuples, duplicate elements are not removed.

    Examples
    --------
    >>> set_remove_subs([(0, 1), (1, 2), (1, 2, 3), (0,)])
    [(1, 2, 3), (0, 1)]
    >>> set_remove_subs([(0, 1), (1, 2), (1,1, 1, 2, 3), (0,)])
    [(1, 1, 1, 2, 3), (0, 1)]

    """
    part = []
    for s in sorted(list(set(ssli)), key=lambda x: len(set(x)))[::-1]:
        if not any((set(s).issubset(set(t)) for t in part)):
            part.append(s)
    return part