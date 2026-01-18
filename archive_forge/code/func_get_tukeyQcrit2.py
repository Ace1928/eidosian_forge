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
def get_tukeyQcrit2(k, df, alpha=0.05):
    """
    return critical values for Tukey's HSD (Q)

    Parameters
    ----------
    k : int in {2, ..., 10}
        number of tests
    df : int
        degrees of freedom of error term
    alpha : {0.05, 0.01}
        type 1 error, 1-confidence level



    not enough error checking for limitations
    """
    return studentized_range.ppf(1 - alpha, k, df)