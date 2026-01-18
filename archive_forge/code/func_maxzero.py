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
def maxzero(x):
    """find all up zero crossings and return the index of the highest

    Not used anymore


    >>> np.random.seed(12345)
    >>> x = np.random.randn(8)
    >>> x
    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057,
            1.39340583,  0.09290788,  0.28174615])
    >>> maxzero(x)
    (4, array([1, 4]))


    no up-zero-crossing at end

    >>> np.random.seed(0)
    >>> x = np.random.randn(8)
    >>> x
    array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
           -0.97727788,  0.95008842, -0.15135721])
    >>> maxzero(x)
    (None, array([6]))
    """
    x = np.asarray(x)
    cond1 = x[:-1] < 0
    cond2 = x[1:] > 0
    allzeros = np.nonzero(cond1 & cond2 | (x[1:] == 0))[0] + 1
    if x[-1] >= 0:
        maxz = max(allzeros)
    else:
        maxz = None
    return (maxz, allzeros)