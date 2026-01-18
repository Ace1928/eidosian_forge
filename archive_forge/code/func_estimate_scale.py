import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings
def estimate_scale(self, Y=None):
    """
        Return Pearson's X^2 estimate of scale.
        """
    if Y is None:
        Y = self.Y
    resid = Y - self.results.mu
    return (np.power(resid, 2) / self.family.variance(self.results.mu)).sum() / self.df_resid