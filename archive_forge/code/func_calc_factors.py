import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
def calc_factors(self, x=None, keepdim=0, addconst=True):
    """get factor decomposition of exogenous variables

        This uses principal component analysis to obtain the factors. The number
        of factors kept is the maximum that will be considered in the regression.
        """
    if x is None:
        x = self.exog
    else:
        x = np.asarray(x)
    xred, fact, evals, evecs = pca(x, keepdim=keepdim, normalize=1)
    self.exog_reduced = xred
    if addconst:
        self.factors = sm.add_constant(fact, prepend=True)
        self.hasconst = 1
    else:
        self.factors = fact
        self.hasconst = 0
    self.evals = evals
    self.evecs = evecs