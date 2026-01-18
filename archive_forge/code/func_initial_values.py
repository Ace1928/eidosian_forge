from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
def initial_values(self, initial_level=None, initial_trend=None, force=False):
    """
        Compute initial values used in the exponential smoothing recursions.

        Parameters
        ----------
        initial_level : {float, None}
            The initial value used for the level component.
        initial_trend : {float, None}
            The initial value used for the trend component.
        force : bool
            Force the calculation even if initial values exist.

        Returns
        -------
        initial_level : float
            The initial value used for the level component.
        initial_trend : {float, None}
            The initial value used for the trend component.
        initial_seasons : list
            The initial values used for the seasonal components.

        Notes
        -----
        Convenience function the exposes the values used to initialize the
        recursions. When optimizing parameters these are used as starting
        values.

        Method used to compute the initial value depends on when components
        are included in the model.  In a simple exponential smoothing model
        without trend or a seasonal components, the initial value is set to the
        first observation. When a trend is added, the trend is initialized
        either using y[1]/y[0], if multiplicative, or y[1]-y[0]. When the
        seasonal component is added the initialization adapts to account for
        the modified structure.
        """
    if self._initialization_method is not None and (not force):
        return (self._initial_level, self._initial_trend, self._initial_seasonal)
    y = self._y
    trend = self.trend
    seasonal = self.seasonal
    has_seasonal = self.has_seasonal
    has_trend = self.has_trend
    m = self.seasonal_periods
    l0 = initial_level
    b0 = initial_trend
    if has_seasonal:
        l0 = y[np.arange(self.nobs) % m == 0].mean() if l0 is None else l0
        if b0 is None and has_trend:
            lead, lag = (y[m:m + m], y[:m])
            if trend == 'mul':
                b0 = np.exp((np.log(lead.mean()) - np.log(lag.mean())) / m)
            else:
                b0 = ((lead - lag) / m).mean()
        s0 = list(y[:m] / l0) if seasonal == 'mul' else list(y[:m] - l0)
    elif has_trend:
        l0 = y[0] if l0 is None else l0
        if b0 is None:
            b0 = y[1] / y[0] if trend == 'mul' else y[1] - y[0]
        s0 = []
    else:
        if l0 is None:
            l0 = y[0]
        b0 = None
        s0 = []
    return (l0, b0, s0)