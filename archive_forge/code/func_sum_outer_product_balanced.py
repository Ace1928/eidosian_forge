import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted
def sum_outer_product_balanced(x, n_groups):
    """sum outerproduct dot(x_i, x_i.T) over individuals

    where x_i is (nobs_i, 1), and result is (nobs_i, nobs_i)

    reshape-dot version, for x.ndim=1 only

    """
    xrs = x.reshape(-1, n_groups, order='F')
    return np.dot(xrs, xrs.T)