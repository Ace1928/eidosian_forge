import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima import specification
def check_methods(spec, order, seasonal_order, enforce_stationarity, enforce_invertibility, concentrate_scale, exog_params, ar_params, ma_params, seasonal_ar_params, seasonal_ma_params, sigma2):
    params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params, seasonal_ma_params, sigma2]
    desired = {'exog_params': exog_params, 'ar_params': ar_params, 'ma_params': ma_params, 'seasonal_ar_params': seasonal_ar_params, 'seasonal_ma_params': seasonal_ma_params}
    if not concentrate_scale:
        desired['sigma2'] = sigma2
    assert_equal(spec.split_params(params), desired)
    assert_equal(spec.join_params(**desired), params)
    assert_equal(spec.validate_params(params), None)
    assert_raises(ValueError, spec.validate_params, [])
    assert_raises(ValueError, spec.validate_params, ['a'] + params[1:].tolist())
    assert_raises(ValueError, spec.validate_params, np.r_[np.inf, params[1:]])
    assert_raises(ValueError, spec.validate_params, np.r_[np.nan, params[1:]])
    if spec.max_ar_order > 0:
        params = np.r_[exog_params, np.ones_like(ar_params), ma_params, np.zeros_like(seasonal_ar_params), seasonal_ma_params, sigma2]
        if enforce_stationarity:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if spec.max_ma_order > 0:
        params = np.r_[exog_params, ar_params, np.ones_like(ma_params), seasonal_ar_params, np.zeros_like(seasonal_ma_params), sigma2]
        if enforce_invertibility:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if spec.max_seasonal_ar_order > 0:
        params = np.r_[exog_params, np.zeros_like(ar_params), ma_params, np.ones_like(seasonal_ar_params), seasonal_ma_params, sigma2]
        if enforce_stationarity:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if spec.max_seasonal_ma_order > 0:
        params = np.r_[exog_params, ar_params, np.zeros_like(ma_params), seasonal_ar_params, np.ones_like(seasonal_ma_params), sigma2]
        if enforce_invertibility:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if not concentrate_scale:
        params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params, seasonal_ma_params, 0.0]
        assert_raises(ValueError, spec.validate_params, params)
        params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params, seasonal_ma_params, -1]
        assert_raises(ValueError, spec.validate_params, params)
    unconstrained_ar_params = ar_params
    unconstrained_ma_params = ma_params
    unconstrained_seasonal_ar_params = seasonal_ar_params
    unconstrained_seasonal_ma_params = seasonal_ma_params
    unconstrained_sigma2 = sigma2
    if spec.max_ar_order > 0 and enforce_stationarity:
        unconstrained_ar_params = unconstrain(np.array(ar_params))
    if spec.max_ma_order > 0 and enforce_invertibility:
        unconstrained_ma_params = unconstrain(-np.array(ma_params))
    if spec.max_seasonal_ar_order > 0 and enforce_stationarity:
        unconstrained_seasonal_ar_params = unconstrain(np.array(seasonal_ar_params))
    if spec.max_seasonal_ma_order > 0 and enforce_invertibility:
        unconstrained_seasonal_ma_params = unconstrain(-np.array(unconstrained_seasonal_ma_params))
    if not concentrate_scale:
        unconstrained_sigma2 = unconstrained_sigma2 ** 0.5
    unconstrained_params = np.r_[exog_params, unconstrained_ar_params, unconstrained_ma_params, unconstrained_seasonal_ar_params, unconstrained_seasonal_ma_params, unconstrained_sigma2]
    params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params, seasonal_ma_params, sigma2]
    assert_allclose(spec.unconstrain_params(params), unconstrained_params)
    assert_allclose(spec.constrain_params(unconstrained_params), params)
    assert_allclose(spec.constrain_params(spec.unconstrain_params(params)), params)