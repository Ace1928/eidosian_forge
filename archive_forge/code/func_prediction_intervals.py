from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.validation import (
from statsmodels.tsa.deterministic import DeterministicTerm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import add_trend, freq_to_period
def prediction_intervals(self, steps: int=1, theta: float=2, alpha: float=0.05) -> pd.DataFrame:
    """
        Parameters
        ----------
        steps : int, default 1
            The number of steps ahead to compute the forecast components.
        theta : float, default 2
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.
        alpha : float, default 0.05
            Significance level for the confidence intervals.

        Returns
        -------
        DataFrame
            DataFrame with columns lower and upper

        Notes
        -----
        The variance of the h-step forecast is assumed to follow from the
        integrated Moving Average structure of the Theta model, and so is
        :math:`\\sigma^2(1 + (h-1)(1 + (\\alpha-1)^2)`. The prediction interval
        assumes that innovations are normally distributed.
        """
    model_alpha = self.params.iloc[1]
    sigma2_h = (1 + np.arange(steps) * (1 + (model_alpha - 1) ** 2)) * self.sigma2
    sigma_h = np.sqrt(sigma2_h)
    quantile = stats.norm.ppf(alpha / 2)
    predictions = self.forecast(steps, theta)
    return pd.DataFrame({'lower': predictions + sigma_h * quantile, 'upper': predictions + sigma_h * -quantile})