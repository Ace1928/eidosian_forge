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
def forecast_components(self, steps: int=1) -> pd.DataFrame:
    """
        Compute the three components of the Theta model forecast

        Parameters
        ----------
        steps : int
            The number of steps ahead to compute the forecast components.

        Returns
        -------
        DataFrame
            A DataFrame with three columns: trend, ses and seasonal containing
            the forecast values of each of the three components.

        Notes
        -----
        For a given value of :math:`\\theta`, the deseasonalized forecast is
        `fcast = w * trend + ses` where :math:`w = \\frac{theta - 1}{theta}`.
        The reseasonalized forecasts are then `seasonal * fcast` if the
        seasonality is multiplicative or `seasonal + fcast` if the seasonality
        is additive.
        """
    steps = int_like(steps, 'steps')
    if steps < 1:
        raise ValueError('steps must be a positive integer')
    alpha = self._alpha
    b0 = self._b0
    nobs = self._nobs
    h = np.arange(1, steps + 1, dtype=np.float64) - 1
    if alpha > 0:
        h += 1 / alpha - (1 - alpha) ** nobs / alpha
    trend = b0 * h
    ses = self._one_step * np.ones(steps)
    if self.model.method.startswith('add'):
        season = np.zeros(steps)
    else:
        season = np.ones(steps)
    if self.model.deseasonalize:
        seasonal = self._seasonal
        period = self.model.period
        oos_idx = nobs + np.arange(steps)
        seasonal_locs = oos_idx % period
        if seasonal.shape[0]:
            season[:] = seasonal[seasonal_locs]
    index = getattr(self.model.endog_orig, 'index', None)
    if index is None:
        index = pd.RangeIndex(0, self.model.endog_orig.shape[0])
    index = extend_index(steps, index)
    df = pd.DataFrame({'trend': trend, 'ses': ses, 'seasonal': season}, index=index)
    return df