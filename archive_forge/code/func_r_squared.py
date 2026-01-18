import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def r_squared(self):
    """
        Returns the R-Squared for the nonparametric regression.

        Notes
        -----
        For more details see p.45 in [2]
        The R-Squared is calculated by:

        .. math:: R^{2}=\\frac{\\left[\\sum_{i=1}^{n}
            (Y_{i}-\\bar{y})(\\hat{Y_{i}}-\\bar{y}\\right]^{2}}{\\sum_{i=1}^{n}
            (Y_{i}-\\bar{y})^{2}\\sum_{i=1}^{n}(\\hat{Y_{i}}-\\bar{y})^{2}},

        where :math:`\\hat{Y_{i}}` is the mean calculated in `fit` at the exog
        points.
        """
    Y = np.squeeze(self.endog)
    Yhat = self.fit()[0]
    Y_bar = np.mean(Yhat)
    R2_numer = ((Y - Y_bar) * (Yhat - Y_bar)).sum() ** 2
    R2_denom = ((Y - Y_bar) ** 2).sum(axis=0) * ((Yhat - Y_bar) ** 2).sum(axis=0)
    return R2_numer / R2_denom