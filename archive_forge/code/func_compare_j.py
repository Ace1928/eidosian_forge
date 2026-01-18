from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def compare_j(self, other):
    """overidentification test for comparing two nested gmm estimates

        This assumes that some moment restrictions have been dropped in one
        of the GMM estimates relative to the other.

        Not tested yet

        We are comparing two separately estimated models, that use different
        weighting matrices. It is not guaranteed that the resulting
        difference is positive.

        TODO: Check in which cases Stata programs use the same weigths

        """
    jstat1 = self.jval
    k_moms1 = self.model.nmoms
    jstat2 = other.jval
    k_moms2 = other.model.nmoms
    jdiff = jstat1 - jstat2
    df = k_moms1 - k_moms2
    if df < 0:
        df = -df
        jdiff = -jdiff
    return (jdiff, stats.chi2.sf(jdiff, df), df)