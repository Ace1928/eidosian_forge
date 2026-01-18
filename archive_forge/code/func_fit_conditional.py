import numpy as np
from scipy import optimize
from statsmodels.regression.linear_model import OLS
def fit_conditional(self, alpha):
    y = self.ar1filter(self.endog, alpha)
    x = self.ar1filter(self.exog, alpha)
    res = OLS(y, x).fit()
    return res.ssr