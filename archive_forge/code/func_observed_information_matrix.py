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
def observed_information_matrix(self, params, transformed=True, includes_fixed=False, approx_complex_step=None, approx_centered=False, **kwargs):
    """
        Observed information matrix

        Parameters
        ----------
        params : array_like, optional
            Array of parameters at which to evaluate the loglikelihood
            function.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

        Notes
        -----
        This method is from Harvey (1989), which shows that the information
        matrix only depends on terms from the gradient. This implementation is
        partially analytic and partially numeric approximation, therefore,
        because it uses the analytic formula for the information matrix, with
        numerically computed elements of the gradient.

        References
        ----------
        Harvey, Andrew C. 1990.
        Forecasting, Structural Time Series Models and the Kalman Filter.
        Cambridge University Press.
        """
    params = np.array(params, ndmin=1)
    n = len(params)
    if approx_complex_step is None:
        approx_complex_step = transformed
    if not transformed and approx_complex_step:
        raise ValueError('Cannot use complex-step approximations to calculate the observed_information_matrix with untransformed parameters.')
    params = self.handle_params(params, transformed=transformed, includes_fixed=includes_fixed)
    self.update(params, transformed=True, includes_fixed=True, complex_step=approx_complex_step)
    if approx_complex_step:
        kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
    res = self.ssm.filter(complex_step=approx_complex_step, **kwargs)
    dtype = self.ssm.dtype
    inv_forecasts_error_cov = res.forecasts_error_cov.copy()
    partials_forecasts_error, partials_forecasts_error_cov = self._forecasts_error_partial_derivatives(params, transformed=transformed, includes_fixed=includes_fixed, approx_complex_step=approx_complex_step, approx_centered=approx_centered, res=res, **kwargs)
    tmp = np.zeros((self.k_endog, self.k_endog, self.nobs, n), dtype=dtype)
    information_matrix = np.zeros((n, n), dtype=dtype)
    d = np.maximum(self.ssm.loglikelihood_burn, res.nobs_diffuse)
    for t in range(d, self.nobs):
        inv_forecasts_error_cov[:, :, t] = np.linalg.inv(res.forecasts_error_cov[:, :, t])
        for i in range(n):
            tmp[:, :, t, i] = np.dot(inv_forecasts_error_cov[:, :, t], partials_forecasts_error_cov[:, :, t, i])
        for i in range(n):
            for j in range(n):
                information_matrix[i, j] += 0.5 * np.trace(np.dot(tmp[:, :, t, i], tmp[:, :, t, j]))
                information_matrix[i, j] += np.inner(partials_forecasts_error[:, t, i], np.dot(inv_forecasts_error_cov[:, :, t], partials_forecasts_error[:, t, j]))
    return information_matrix / (self.nobs - self.ssm.loglikelihood_burn)