from pandas
import numpy as np
from statsmodels.regression.linear_model import GLS, RegressionResults
@property
def rnorm_cov_params(self):
    """Parameter covariance under restrictions"""
    if self._ncp is None:
        P = self.ncoeffs
        self._ncp = self.inv_rwexog[:P, :P]
    return self._ncp