from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def forecast_interval(self, y, steps, alpha=0.05, exog_future=None):
    """
        Construct forecast interval estimates assuming the y are Gaussian

        Parameters
        ----------
        y : {ndarray, None}
            The initial values to use for the forecasts. If None,
            the last k_ar values of the original endogenous variables are
            used.
        steps : int
            Number of steps ahead to forecast
        alpha : float, optional
            The significance level for the confidence intervals.
        exog_future : ndarray, optional
            Forecast values of the exogenous variables. Should include
            constant, trend, etc. as needed, including extrapolating out
            of sample.
        Returns
        -------
        point : ndarray
            Mean value of forecast
        lower : ndarray
            Lower bound of confidence interval
        upper : ndarray
            Upper bound of confidence interval

        Notes
        -----
        Lütkepohl pp. 39-40
        """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    q = util.norm_signif_level(alpha)
    point_forecast = self.forecast(y, steps, exog_future=exog_future)
    sigma = np.sqrt(self._forecast_vars(steps))
    forc_lower = point_forecast - q * sigma
    forc_upper = point_forecast + q * sigma
    return (point_forecast, forc_lower, forc_upper)