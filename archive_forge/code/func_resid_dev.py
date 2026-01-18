import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def resid_dev(self, endog, mu, var_weights=1.0, scale=1.0):
    """
        The deviance residuals

        Parameters
        ----------
        endog : array_like
            The endogenous response variable
        mu : array_like
            The inverse of the link function at the linear predicted values.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is 1.
        scale : float, optional
            An optional scale argument. The default is 1.

        Returns
        -------
        resid_dev : float
            Deviance residuals as defined below.

        Notes
        -----
        The deviance residuals are defined by the contribution D_i of
        observation i to the deviance as

        .. math::
           resid\\_dev_i = sign(y_i-\\mu_i) \\sqrt{D_i}

        D_i is calculated from the _resid_dev method in each family.
        Distribution-specific documentation of the calculation is available
        there.
        """
    resid_dev = self._resid_dev(endog, mu)
    resid_dev *= var_weights / scale
    return np.sign(endog - mu) * np.sqrt(np.clip(resid_dev, 0.0, np.inf))