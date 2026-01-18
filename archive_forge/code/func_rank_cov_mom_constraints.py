import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
@cache_readonly
def rank_cov_mom_constraints(self):
    return np.linalg.matrix_rank(self.cov_mom_constraints)