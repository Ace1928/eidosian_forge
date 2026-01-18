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
def multicontrast_pvalues(tstat, tcorr, df=None, dist='t', alternative='two-sided'):
    """pvalues for simultaneous tests

    """
    from statsmodels.sandbox.distributions.multivariate import mvstdtprob
    if df is None and dist == 't':
        raise ValueError('df has to be specified for the t-distribution')
    tstat = np.asarray(tstat)
    ntests = len(tstat)
    cc = np.abs(tstat)
    pval_global = 1 - mvstdtprob(-cc, cc, tcorr, df)
    pvals = []
    for ti in cc:
        limits = ti * np.ones(ntests)
        pvals.append(1 - mvstdtprob(-cc, cc, tcorr, df))
    return (pval_global, np.asarray(pvals))