from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_LM_homoskedastic(self):
    resid = self.res1_restricted.wresid
    n = resid.shape[0]
    x = self.x
    S = np.dot(resid, resid) / n * np.dot(x.T, x) / n
    Sinv = np.linalg.inv(S)
    s = np.mean(x * resid[:, None], 0)
    LMstat = n * np.dot(np.dot(s, Sinv), s.T)
    LMstat_OLS = self.res1_full.compare_lm_test(self.res1_restricted)
    LMstat2 = LMstat_OLS[0]
    assert_almost_equal(LMstat, LMstat2, DECIMAL_7)