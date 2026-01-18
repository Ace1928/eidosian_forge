import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def fitted(self, lin_pred):
    """
        Fitted values based on linear predictors lin_pred.

        Parameters
        ----------
        lin_pred : ndarray
            Values of the linear predictor of the model.
            :math:`X \\cdot \\beta` in a classical linear model.

        Returns
        -------
        mu : ndarray
            The mean response variables given by the inverse of the link
            function.
        """
    fits = self.link.inverse(lin_pred)
    return fits