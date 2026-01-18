from pandas
import numpy as np
from statsmodels.regression.linear_model import GLS, RegressionResults
@property
def rwendog(self):
    """Whitened endogenous variable augmented with restriction parameters"""
    if self._rwendog is None:
        P = self.ncoeffs
        K = self.nconstraint
        response = np.zeros((P + K,))
        response[:P] = np.dot(self.wexog.T, self.wendog)
        response[P:] = self.param
        self._rwendog = response
    return self._rwendog