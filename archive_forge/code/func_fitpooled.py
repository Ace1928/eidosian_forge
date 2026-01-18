import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
def fitpooled(self):
    """fit the pooled model, which assumes there are no differences across groups
        """
    if self.het:
        if not hasattr(self, 'weights'):
            self.fitbygroups()
        weights = self.weights
        res = WLS(self.endog, self.exog, weights=weights).fit()
    else:
        res = OLS(self.endog, self.exog).fit()
    self.lspooled = res