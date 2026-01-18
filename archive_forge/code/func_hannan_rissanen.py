import numpy as np
from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
def hannan_rissanen(endog, ar_order=0, ma_order=0, demean=True, initial_ar_order=None, unbiased=None, fixed_params=None):
    """
    Estimate ARMA parameters using Hannan-Rissanen procedure.

    Parameters
    ----------
    endog : array_like
        Input time series array, assumed to be stationary.
    ar_order : int or list of int
        Autoregressive order
    ma_order : int or list of int
        Moving average order
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the ARMA coefficients. Default is True.
    initial_ar_order : int, optional
        Order of long autoregressive process used for initial computation of
        residuals.
    unbiased : bool, optional
        Whether or not to apply the bias correction step. Default is True if
        the estimated coefficients from the previous step imply a stationary
        and invertible process and False otherwise.
    fixed_params : dict, optional
        Dictionary with names of fixed parameters as keys (e.g. 'ar.L1',
        'ma.L2'), which correspond to SARIMAXSpecification.param_names.
        Dictionary values are the values of the associated fixed parameters.

    Returns
    -------
    parameters : SARIMAXParams object
    other_results : Bunch
        Includes three components: `spec`, containing the
        `SARIMAXSpecification` instance corresponding to the input arguments;
        `initial_ar_order`, containing the autoregressive lag order used in the
        first step; and `resid`, which contains the computed residuals from the
        last step.

    Notes
    -----
    The primary reference is [1]_, section 5.1.4, which describes a three-step
    procedure that we implement here.

    1. Fit a large-order AR model via Yule-Walker to estimate residuals
    2. Compute AR and MA estimates via least squares
    3. (Unless the estimated coefficients from step (2) are non-stationary /
       non-invertible or `unbiased=False`) Perform bias correction

    The order used for the AR model in the first step may be given as an
    argument. If it is not, we compute it as suggested by [2]_.

    The estimate of the variance that we use is computed from the residuals
    of the least-squares regression and not from the innovations algorithm.
    This is because our fast implementation of the innovations algorithm is
    only valid for stationary processes, and the Hannan-Rissanen procedure may
    produce estimates that imply non-stationary processes. To avoid
    inconsistency, we never compute this latter variance here, even if it is
    possible. See test_hannan_rissanen::test_brockwell_davis_example_517 for
    an example of how to compute this variance manually.

    This procedure assumes that the series is stationary, but if this is not
    true, it is still possible that this procedure will return parameters that
    imply a non-stationary / non-invertible process.

    Note that the third stage will only be applied if the parameters from the
    second stage imply a stationary / invertible model. If `unbiased=True` is
    given, then non-stationary / non-invertible parameters in the second stage
    will throw an exception.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    .. [2] Gomez, Victor, and Agustin Maravall. 2001.
       "Automatic Modeling Methods for Univariate Series."
       A Course in Time Series Analysis, 171–201.
    """
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)
    fixed_params = _validate_fixed_params(fixed_params, spec.param_names)
    endog = spec.endog
    if demean:
        endog = endog - endog.mean()
    p = SARIMAXParams(spec=spec)
    nobs = len(endog)
    max_ar_order = spec.max_ar_order
    max_ma_order = spec.max_ma_order
    if initial_ar_order is None:
        initial_ar_order = max(np.floor(np.log(nobs) ** 2).astype(int), 2 * max(max_ar_order, max_ma_order))
    _ = SARIMAXSpecification(endog, ar_order=initial_ar_order)
    params_info = _package_fixed_and_free_params_info(fixed_params, spec.ar_lags, spec.ma_lags)
    lagged_endog = lagmat(endog, max_ar_order, trim='both')
    mod = None
    if max_ma_order == 0 and max_ar_order == 0:
        p.sigma2 = np.var(endog, ddof=0)
        resid = endog.copy()
    elif max_ma_order == 0:
        X_with_free_params = lagged_endog[:, params_info.free_ar_ix]
        X_with_fixed_params = lagged_endog[:, params_info.fixed_ar_ix]
        y = endog[max_ar_order:]
        if X_with_fixed_params.shape[1] != 0:
            y = y - X_with_fixed_params.dot(params_info.fixed_ar_params)
        if X_with_free_params.shape[1] == 0:
            p.ar_params = params_info.fixed_ar_params
            p.sigma2 = np.var(y, ddof=0)
            resid = y.copy()
        else:
            mod = OLS(y, X_with_free_params)
            res = mod.fit()
            resid = res.resid
            p.sigma2 = res.scale
            p.ar_params = _stitch_fixed_and_free_params(fixed_ar_or_ma_lags=params_info.fixed_ar_lags, fixed_ar_or_ma_params=params_info.fixed_ar_params, free_ar_or_ma_lags=params_info.free_ar_lags, free_ar_or_ma_params=res.params, spec_ar_or_ma_lags=spec.ar_lags)
    else:
        initial_ar_params, _ = yule_walker(endog, order=initial_ar_order, method='mle')
        X = lagmat(endog, initial_ar_order, trim='both')
        y = endog[initial_ar_order:]
        resid = y - X.dot(initial_ar_params)
        lagged_resid = lagmat(resid, max_ma_order, trim='both')
        ix = initial_ar_order + max_ma_order - max_ar_order
        X_with_free_params = np.c_[lagged_endog[ix:, params_info.free_ar_ix], lagged_resid[:, params_info.free_ma_ix]]
        X_with_fixed_params = np.c_[lagged_endog[ix:, params_info.fixed_ar_ix], lagged_resid[:, params_info.fixed_ma_ix]]
        y = endog[initial_ar_order + max_ma_order:]
        if X_with_fixed_params.shape[1] != 0:
            y = y - X_with_fixed_params.dot(np.r_[params_info.fixed_ar_params, params_info.fixed_ma_params])
        if X_with_free_params.shape[1] == 0:
            p.ar_params = params_info.fixed_ar_params
            p.ma_params = params_info.fixed_ma_params
            p.sigma2 = np.var(y, ddof=0)
            resid = y.copy()
        else:
            mod = OLS(y, X_with_free_params)
            res = mod.fit()
            k_free_ar_params = len(params_info.free_ar_lags)
            p.ar_params = _stitch_fixed_and_free_params(fixed_ar_or_ma_lags=params_info.fixed_ar_lags, fixed_ar_or_ma_params=params_info.fixed_ar_params, free_ar_or_ma_lags=params_info.free_ar_lags, free_ar_or_ma_params=res.params[:k_free_ar_params], spec_ar_or_ma_lags=spec.ar_lags)
            p.ma_params = _stitch_fixed_and_free_params(fixed_ar_or_ma_lags=params_info.fixed_ma_lags, fixed_ar_or_ma_params=params_info.fixed_ma_params, free_ar_or_ma_lags=params_info.free_ma_lags, free_ar_or_ma_params=res.params[k_free_ar_params:], spec_ar_or_ma_lags=spec.ma_lags)
            resid = res.resid
            p.sigma2 = res.scale
        if unbiased is True:
            if len(fixed_params) != 0:
                raise NotImplementedError('Third step of Hannan-Rissanen estimation to remove parameter bias is not yet implemented for the case with fixed parameters.')
            elif not (p.is_stationary and p.is_invertible):
                raise ValueError('Cannot perform third step of Hannan-Rissanen estimation to remove parameter bias, because parameters estimated from the second step are non-stationary or non-invertible.')
        elif unbiased is None:
            if len(fixed_params) != 0:
                unbiased = False
            else:
                unbiased = p.is_stationary and p.is_invertible
        if unbiased is True:
            if mod is None:
                raise ValueError('Must have free parameters to use unbiased')
            Z = np.zeros_like(endog)
            ar_coef = p.ar_poly.coef
            ma_coef = p.ma_poly.coef
            for t in range(nobs):
                if t >= max(max_ar_order, max_ma_order):
                    tmp_ar = np.dot(-ar_coef[1:], endog[t - max_ar_order:t][::-1])
                    tmp_ma = np.dot(ma_coef[1:], Z[t - max_ma_order:t][::-1])
                    Z[t] = endog[t] - tmp_ar - tmp_ma
            V = lfilter([1], ar_coef, Z)
            W = lfilter(np.r_[1, -ma_coef[1:]], [1], Z)
            lagged_V = lagmat(V, max_ar_order, trim='both')
            lagged_W = lagmat(W, max_ma_order, trim='both')
            exog = np.c_[lagged_V[max(max_ma_order - max_ar_order, 0):, params_info.free_ar_ix], lagged_W[max(max_ar_order - max_ma_order, 0):, params_info.free_ma_ix]]
            mod_unbias = OLS(Z[max(max_ar_order, max_ma_order):], exog)
            res_unbias = mod_unbias.fit()
            p.ar_params = p.ar_params + res_unbias.params[:spec.k_ar_params]
            p.ma_params = p.ma_params + res_unbias.params[spec.k_ar_params:]
            resid = mod.endog - mod.exog.dot(np.r_[p.ar_params, p.ma_params])
            p.sigma2 = np.inner(resid, resid) / len(resid)
    other_results = Bunch({'spec': spec, 'initial_ar_order': initial_ar_order, 'resid': resid})
    return (p, other_results)