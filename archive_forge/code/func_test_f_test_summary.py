import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
@pytest.mark.smoke
def test_f_test_summary(self):
    res1 = self.res1
    mat = np.eye(len(res1.params))
    ft = res1.f_test(mat[:-1], cov_p=self.cov_robust)
    ft.summary()