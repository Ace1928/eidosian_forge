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
def fdrcorrection_bak(pvals, alpha=0.05, method='indep'):
    """Reject False discovery rate correction for pvalues

    Old version, to be deleted


    missing: methods that estimate fraction of true hypotheses

    """
    pvals = np.asarray(pvals)
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    pecdf = ecdf(pvals_sorted)
    if method in ['i', 'indep', 'p', 'poscorr']:
        rline = pvals_sorted / alpha
    elif method in ['n', 'negcorr']:
        cm = np.sum(1.0 / np.arange(1, len(pvals)))
        rline = pvals_sorted / alpha * cm
    elif method in ['g', 'onegcorr']:
        rline = pvals_sorted / (pvals_sorted * (1 - alpha) + alpha)
    elif method in ['oth', 'o2negcorr']:
        cm = np.sum(np.arange(len(pvals)))
        rline = pvals_sorted / alpha / cm
    else:
        raise ValueError('method not available')
    reject = pecdf >= rline
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True
    return reject[pvals_sortind.argsort()]