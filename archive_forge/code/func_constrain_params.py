import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
def constrain_params(self, unconstrained):
    """
        Constrain parameter values to be valid through transformations.

        Parameters
        ----------
        unconstrained : array_like
            Array of model unconstrained parameters.

        Returns
        -------
        constrained : ndarray
            Array of model parameters transformed to produce a valid model.

        Notes
        -----
        This is usually only used when performing numerical minimization
        of the log-likelihood function. This function is necessary because
        the minimizers consider values over the entire real space, while
        SARIMAX models require parameters in subspaces (for example positive
        variances).

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.constrain_params([10, -2])
        array([-0.99504,  4.     ])
        """
    unconstrained = self.split_params(unconstrained)
    params = {}
    if self.k_exog_params:
        params['exog_params'] = unconstrained['exog_params']
    if self.k_ar_params:
        if self.enforce_stationarity:
            params['ar_params'] = constrain(unconstrained['ar_params'])
        else:
            params['ar_params'] = unconstrained['ar_params']
    if self.k_ma_params:
        if self.enforce_invertibility:
            params['ma_params'] = -constrain(unconstrained['ma_params'])
        else:
            params['ma_params'] = unconstrained['ma_params']
    if self.k_seasonal_ar_params:
        if self.enforce_stationarity:
            params['seasonal_ar_params'] = constrain(unconstrained['seasonal_ar_params'])
        else:
            params['seasonal_ar_params'] = unconstrained['seasonal_ar_params']
    if self.k_seasonal_ma_params:
        if self.enforce_invertibility:
            params['seasonal_ma_params'] = -constrain(unconstrained['seasonal_ma_params'])
        else:
            params['seasonal_ma_params'] = unconstrained['seasonal_ma_params']
    if not self.concentrate_scale:
        params['sigma2'] = unconstrained['sigma2'] ** 2
    return self.join_params(**params)