import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
def join_params(self, exog_params=None, ar_params=None, ma_params=None, seasonal_ar_params=None, seasonal_ma_params=None, sigma2=None):
    """
        Join parameters into a single vector.

        Parameters
        ----------
        exog_params : array_like, optional
            Parameters associated with exogenous regressors. Required if
            `exog` is part of specification.
        ar_params : array_like, optional
            Parameters associated with (non-seasonal) autoregressive component.
            Required if this component is part of the specification.
        ma_params : array_like, optional
            Parameters associated with (non-seasonal) moving average component.
            Required if this component is part of the specification.
        seasonal_ar_params : array_like, optional
            Parameters associated with seasonal autoregressive component.
            Required if this component is part of the specification.
        seasonal_ma_params : array_like, optional
            Parameters associated with seasonal moving average component.
            Required if this component is part of the specification.
        sigma2 : array_like, optional
            Innovation variance parameter. Required unless
            `concentrated_scale=True`.

        Returns
        -------
        params : ndarray
            Array of parameters.

        Examples
        --------
        >>> spec = SARIMAXSpecification(ar_order=1)
        >>> spec.join_params(ar_params=0.5, sigma2=4)
        array([0.5, 4. ])
        """
    definitions = [('exogenous variables', self.k_exog_params, exog_params), ('AR terms', self.k_ar_params, ar_params), ('MA terms', self.k_ma_params, ma_params), ('seasonal AR terms', self.k_seasonal_ar_params, seasonal_ar_params), ('seasonal MA terms', self.k_seasonal_ma_params, seasonal_ma_params), ('variance', int(not self.concentrate_scale), sigma2)]
    params_list = []
    for title, k, params in definitions:
        if k > 0:
            if params is None:
                raise ValueError('Specification includes %s, but no parameters were provided.' % title)
            params = np.atleast_1d(np.squeeze(params))
            if not params.shape == (k,):
                raise ValueError('Specification included %d %s, but parameters with shape %s were provided.' % (k, title, params.shape))
            params_list.append(params)
    return np.concatenate(params_list)