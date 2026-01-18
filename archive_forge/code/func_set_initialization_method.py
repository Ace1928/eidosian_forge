from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
def set_initialization_method(self, initialization_method, initial_level=None, initial_trend=None, initial_seasonal=None):
    """
        Sets a new initialization method for the state space model.

        Parameters
        ----------
        initialization_method : str, optional
            Method for initialization of the state space model. One of:

            * 'estimated' (default)
            * 'heuristic'
            * 'known'

            If 'known' initialization is used, then `initial_level` must be
            passed, as well as `initial_trend` and `initial_seasonal` if
            applicable.
            'heuristic' uses a heuristic based on the data to estimate initial
            level, trend, and seasonal state. 'estimated' uses the same
            heuristic as initial guesses, but then estimates the initial states
            as part of the fitting process. Default is 'estimated'.
        initial_level : float, optional
            The initial level component. Only used if initialization is
            'known'.
        initial_trend : float, optional
            The initial trend component. Only used if initialization is
            'known'.
        initial_seasonal : array_like, optional
            The initial seasonal component. An array of length
            `seasonal_periods`. Only used if initialization is 'known'.
        """
    self.initialization_method = string_like(initialization_method, 'initialization_method', options=('estimated', 'known', 'heuristic'))
    if self.initialization_method == 'known':
        if initial_level is None:
            raise ValueError('`initial_level` argument must be provided when initialization method is set to "known".')
        if self.has_trend and initial_trend is None:
            raise ValueError('`initial_trend` argument must be provided for models with a trend component when initialization method is set to "known".')
        if self.has_seasonal and initial_seasonal is None:
            raise ValueError('`initial_seasonal` argument must be provided for models with a seasonal component when initialization method is set to "known".')
    elif self.initialization_method == 'heuristic':
        initial_level, initial_trend, initial_seasonal = _initialization_heuristic(self.endog, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
    elif self.initialization_method == 'estimated':
        if self.nobs < 10 + 2 * (self.seasonal_periods // 2):
            initial_level, initial_trend, initial_seasonal = _initialization_simple(self.endog, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
        else:
            initial_level, initial_trend, initial_seasonal = _initialization_heuristic(self.endog, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
    if not self.has_trend:
        initial_trend = 0
    if not self.has_seasonal:
        initial_seasonal = 0
    self.initial_level = initial_level
    self.initial_trend = initial_trend
    self.initial_seasonal = initial_seasonal
    self._internal_params_index = OrderedDict(zip(self._internal_param_names, np.arange(self._k_params_internal)))
    self._params_index = OrderedDict(zip(self.param_names, np.arange(self.k_params)))