import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
@cache_readonly
def asy_transform_mom_constraints(self):
    res = self.transf_mt.dot(self.project_w)
    return res