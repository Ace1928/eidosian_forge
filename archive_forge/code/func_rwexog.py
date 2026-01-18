from pandas
import numpy as np
from statsmodels.regression.linear_model import GLS, RegressionResults
@property
def rwexog(self):
    """Whitened exogenous variables augmented with restrictions"""
    if self._rwexog is None:
        P = self.ncoeffs
        K = self.nconstraint
        design = np.zeros((P + K, P + K))
        design[:P, :P] = np.dot(self.wexog.T, self.wexog)
        constr = np.reshape(self.constraint, (K, P))
        design[:P, P:] = constr.T
        design[P:, :P] = constr
        design[P:, P:] = np.zeros((K, K))
        self._rwexog = design
    return self._rwexog