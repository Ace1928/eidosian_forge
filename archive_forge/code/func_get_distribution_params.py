import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def get_distribution_params(self, exog=None, exog_precision=None, transform=True):
    """
        Return distribution parameters converted from model prediction.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        transform : bool
            If transform is True and formulas have been used, then predictor
            ``exog`` is passed through the formula processing. Default is True.

        Returns
        -------
        (alpha, beta) : tuple of ndarrays
            Parameters for the scipy distribution to evaluate predictive
            distribution.
        """
    mean = self.predict(exog=exog, transform=transform)
    precision = self.predict(exog_precision=exog_precision, which='precision', transform=transform)
    return (precision * mean, precision * (1 - mean))