from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def get_forecast(self, steps=1, signal_only=False, **kwargs):
    """
        Out-of-sample forecasts and prediction intervals

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default is 1.
        signal_only : bool, optional
            Whether to compute forecasts of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \\alpha_t + \\varepsilon_t`, and the "signal"
            component is then :math:`Z \\alpha_t`. If this argument is set to
            True, then forecasts of the "signal" :math:`Z \\alpha_t` will be
            returned. Otherwise, the default is for forecasts of :math:`y_t`
            to be returned.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecasts : PredictionResults
            PredictionResults instance containing out-of-sample forecasts and
            results including confidence intervals.

        See also
        --------
        forecast
            Out-of-sample forecasts.
        predict
            In-sample predictions and out-of-sample forecasts.
        get_prediction
            In-sample predictions / out-of-sample forecasts and results
            including confidence intervals.
        """
    if isinstance(steps, int):
        end = self.nobs + steps - 1
    else:
        end = steps
    return self.get_prediction(start=self.nobs, end=end, signal_only=signal_only, **kwargs)