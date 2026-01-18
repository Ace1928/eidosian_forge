import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def hessian_factor(self, params, observed=True):
    """Derivatives of loglikelihood function w.r.t. linear predictors.
        """
    _, hf = self.score_hessian_factor(params, return_hessian=True, observed=observed)
    return hf