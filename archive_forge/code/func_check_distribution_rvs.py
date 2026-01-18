import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def check_distribution_rvs(dist, args, alpha, rvs):
    D, pval = stats.kstest(rvs, dist, args=args, N=1000)
    if pval < alpha:
        D, pval = stats.kstest(dist, dist, args=args, N=1000)
        npt.assert_(pval > alpha, 'D = ' + str(D) + '; pval = ' + str(pval) + '; alpha = ' + str(alpha) + '\nargs = ' + str(args))