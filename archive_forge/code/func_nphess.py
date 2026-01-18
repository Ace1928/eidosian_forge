import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def nphess(params, model):
    nobs = model.nobs
    pen_hess = alpha[k] * (1 - L1_wt)
    h = -model.hessian(np.r_[params], **hess_kwds)[0, 0] / nobs + pen_hess
    return h