import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
def nloglikeobs(self, params):
    if self.fixed_params is not None:
        params = self.expandparams(params)
    b = params[0]
    loc = params[1]
    scale = params[2]
    endog = self.endog
    x = (endog - loc) / scale
    logpdf = np.log(b) - (b + 1.0) * np.log(x)
    logpdf -= np.log(scale)
    logpdf[x < 1] = -10000
    return -logpdf