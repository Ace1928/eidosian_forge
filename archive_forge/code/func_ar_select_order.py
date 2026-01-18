from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
@Substitution(auto_reg_params=_auto_reg_params)
def ar_select_order(endog, maxlag, ic='bic', glob=False, trend: Literal['n', 'c', 'ct', 'ctt']='c', seasonal=False, exog=None, hold_back=None, period=None, missing='none', old_names=False):
    """
    Autoregressive AR-X(p) model order selection.

    Parameters
    ----------
    endog : array_like
         A 1-d endogenous response variable. The independent variable.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}
        The information criterion to use in the selection.
    glob : bool
        Flag indicating where to use a global search  across all combinations
        of lags.  In practice, this option is not computational feasible when
        maxlag is larger than 15 (or perhaps 20) since the global search
        requires fitting 2**maxlag models.
%(auto_reg_params)s

    Returns
    -------
    AROrderSelectionResults
        A results holder containing the model and the complete set of
        information criteria for all models fit.

    Examples
    --------
    >>> from statsmodels.tsa.ar_model import ar_select_order
    >>> data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']

    Determine the optimal lag structure

    >>> mod = ar_select_order(data, maxlag=13)
    >>> mod.ar_lags
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    Determine the optimal lag structure with seasonal terms

    >>> mod = ar_select_order(data, maxlag=13, seasonal=True, period=12)
    >>> mod.ar_lags
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    Globally determine the optimal lag structure

    >>> mod = ar_select_order(data, maxlag=13, glob=True)
    >>> mod.ar_lags
    array([1, 2, 9])
    """
    full_mod = AutoReg(endog, maxlag, trend=trend, seasonal=seasonal, exog=exog, hold_back=hold_back, period=period, missing=missing, old_names=old_names)
    nexog = full_mod.exog.shape[1] if full_mod.exog is not None else 0
    y, x = (full_mod._y, full_mod._x)
    base_col = x.shape[1] - nexog - maxlag
    sel = np.ones(x.shape[1], dtype=bool)
    ics: list[tuple[int | tuple[int, ...], tuple[float, float, float]]] = []

    def compute_ics(res):
        nobs = res.nobs
        df_model = res.df_model
        sigma2 = 1.0 / nobs * sumofsq(res.resid)
        llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
        res = SimpleNamespace(nobs=nobs, df_model=df_model, sigma2=sigma2, llf=llf)
        aic = call_cached_func(AutoRegResults.aic, res)
        bic = call_cached_func(AutoRegResults.bic, res)
        hqic = call_cached_func(AutoRegResults.hqic, res)
        return (aic, bic, hqic)

    def ic_no_data():
        """Fake mod and results to handle no regressor case"""
        mod = SimpleNamespace(nobs=y.shape[0], endog=y, exog=np.empty((y.shape[0], 0)))
        llf = OLS.loglike(mod, np.empty(0))
        res = SimpleNamespace(resid=y, nobs=y.shape[0], llf=llf, df_model=0, k_constant=0)
        return compute_ics(res)
    if not glob:
        sel[base_col:base_col + maxlag] = False
        for i in range(maxlag + 1):
            sel[base_col:base_col + i] = True
            if not np.any(sel):
                ics.append((0, ic_no_data()))
                continue
            res = OLS(y, x[:, sel]).fit()
            lags = tuple((j for j in range(1, i + 1)))
            lags = 0 if not lags else lags
            ics.append((lags, compute_ics(res)))
    else:
        bits = np.arange(2 ** maxlag, dtype=np.int32)[:, None]
        bits = bits.view(np.uint8)
        bits = np.unpackbits(bits).reshape(-1, 32)
        for i in range(4):
            bits[:, 8 * i:8 * (i + 1)] = bits[:, 8 * i:8 * (i + 1)][:, ::-1]
        masks = bits[:, :maxlag]
        for mask in masks:
            sel[base_col:base_col + maxlag] = mask
            if not np.any(sel):
                ics.append((0, ic_no_data()))
                continue
            res = OLS(y, x[:, sel]).fit()
            lags = tuple(np.where(mask)[0] + 1)
            lags = 0 if not lags else lags
            ics.append((lags, compute_ics(res)))
    key_loc = {'aic': 0, 'bic': 1, 'hqic': 2}[ic]
    ics = sorted(ics, key=lambda x: x[1][key_loc])
    selected_model = ics[0][0]
    mod = AutoReg(endog, selected_model, trend=trend, seasonal=seasonal, exog=exog, hold_back=hold_back, period=period, missing=missing, old_names=old_names)
    return AROrderSelectionResults(mod, ics, trend, seasonal, period)