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
def test_LM_heteroskedastic_LRversion(self):
    resid = self.res1_restricted.wresid
    resid_full = self.res1_full.wresid
    n = resid.shape[0]
    x = self.x
    scores = x * resid[:, None]
    s = np.mean(scores, 0)
    scores = x * resid_full[:, None]
    S = np.dot(scores.T, scores) / n
    Sinv = np.linalg.inv(S)
    LMstat = n * np.dot(np.dot(s, Sinv), s.T)
    LMstat_OLS = self.res2_full.compare_lm_test(self.res2_restricted, use_lr=True)
    LMstat2 = LMstat_OLS[0]
    assert_almost_equal(LMstat, LMstat2, DECIMAL_7)