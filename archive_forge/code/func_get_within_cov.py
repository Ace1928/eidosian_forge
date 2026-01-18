import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def get_within_cov(self, resid):
    mom = sum_outer_product_loop(resid, self.group.group_iter)
    return mom / self.n_groups