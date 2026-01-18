from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
def ardl_select_order(endog: ArrayLike1D | ArrayLike2D, maxlag: int, exog: ArrayLike2D, maxorder: int | dict[Hashable, int], trend: Literal['n', 'c', 'ct', 'ctt']='c', *, fixed: ArrayLike2D | None=None, causal: bool=False, ic: Literal['aic', 'bic']='bic', glob: bool=False, seasonal: bool=False, deterministic: DeterministicProcess | None=None, hold_back: int | None=None, period: int | None=None, missing: Literal['none', 'raise']='none') -> ARDLOrderSelectionResults:
    """
    ARDL order selection

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    maxlag : int
        The maximum lag to consider for the endogenous variable.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    maxorder : {int, dict}
        If int, sets a common max lag length for all exog variables. If
        a dict, then sets individual lag length. They keys are column names
        if exog is a DataFrame or column indices otherwise.
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    ic : {"aic", "bic", "hqic"}
        The information criterion to use in model selection.
    glob : bool
        Whether to consider all possible submodels of the largest model
        or only if smaller order lags must be included if larger order
        lags are.  If ``True``, the number of model considered is of the
        order 2**(maxlag + k * maxorder) assuming maxorder is an int. This
        can be very large unless k and maxorder are bot relatively small.
        If False, the number of model considered is of the order
        maxlag*maxorder**k which may also be substantial when k and maxorder
        are large.
    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Returns
    -------
    ARDLSelectionResults
        A results holder containing the selected model and the complete set
        of information criteria for all models fit.
    """
    orig_hold_back = int_like(hold_back, 'hold_back', optional=True)

    def compute_ics(y, x, df):
        if x.shape[1]:
            resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        else:
            resid = y
        nobs = resid.shape[0]
        sigma2 = 1.0 / nobs * sumofsq(resid)
        llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
        res = SimpleNamespace(nobs=nobs, df_model=df + x.shape[1], sigma2=sigma2, llf=llf)
        aic = call_cached_func(ARDLResults.aic, res)
        bic = call_cached_func(ARDLResults.bic, res)
        hqic = call_cached_func(ARDLResults.hqic, res)
        return (aic, bic, hqic)
    base = ARDL(endog, maxlag, exog, maxorder, trend, fixed=fixed, causal=causal, seasonal=seasonal, deterministic=deterministic, hold_back=hold_back, period=period, missing=missing)
    hold_back = base.hold_back
    blocks = base._blocks
    always = np.column_stack([blocks['deterministic'], blocks['fixed']])
    always = always[hold_back:]
    select = []
    iter_orders = []
    select.append(blocks['endog'][hold_back:])
    iter_orders.append(list(range(blocks['endog'].shape[1] + 1)))
    var_names = []
    for var in blocks['exog']:
        block = blocks['exog'][var][hold_back:]
        select.append(block)
        iter_orders.append(list(range(block.shape[1] + 1)))
        var_names.append(var)
    y = base._y
    if always.shape[1]:
        pinv_always = np.linalg.pinv(always)
        for i in range(len(select)):
            x = select[i]
            select[i] = x - always @ (pinv_always @ x)
        y = y - always @ (pinv_always @ y)

    def perm_to_tuple(keys, perm):
        if perm == ():
            d = {k: 0 for k, _ in keys if k is not None}
            return (0,) + tuple(((k, v) for k, v in d.items()))
        d = defaultdict(list)
        y_lags = []
        for v in perm:
            key = keys[v]
            if key[0] is None:
                y_lags.append(key[1])
            else:
                d[key[0]].append(key[1])
        d = dict(d)
        if not y_lags or y_lags == [0]:
            y_lags = 0
        else:
            y_lags = tuple(y_lags)
        for key in keys:
            if key[0] not in d and key[0] is not None:
                d[key[0]] = None
        for key in d:
            if d[key] is not None:
                d[key] = tuple(d[key])
        return (y_lags,) + tuple(((k, v) for k, v in d.items()))
    always_df = always.shape[1]
    ics = {}
    if glob:
        ar_lags = base.ar_lags if base.ar_lags is not None else []
        keys = [(None, i) for i in ar_lags]
        for k, v in base._order.items():
            keys += [(k, i) for i in v]
        x = np.column_stack([a for a in select])
        all_columns = list(range(x.shape[1]))
        for i in range(x.shape[1]):
            for perm in combinations(all_columns, i):
                key = perm_to_tuple(keys, perm)
                ics[key] = compute_ics(y, x[:, perm], always_df)
    else:
        for io in product(*iter_orders):
            x = np.column_stack([a[:, :io[i]] for i, a in enumerate(select)])
            key = [io[0] if io[0] else None]
            for j, val in enumerate(io[1:]):
                var = var_names[j]
                if causal:
                    key.append((var, None if val == 0 else val))
                else:
                    key.append((var, val - 1 if val - 1 >= 0 else None))
            key = tuple(key)
            ics[key] = compute_ics(y, x, always_df)
    index = {'aic': 0, 'bic': 1, 'hqic': 2}[ic]
    lowest = np.inf
    for key in ics:
        val = ics[key][index]
        if val < lowest:
            lowest = val
            selected_order = key
    exog_order = {k: v for k, v in selected_order[1:]}
    model = ARDL(endog, selected_order[0], exog, exog_order, trend, fixed=fixed, causal=causal, seasonal=seasonal, deterministic=deterministic, hold_back=orig_hold_back, period=period, missing=missing)
    return ARDLOrderSelectionResults(model, ics, trend, seasonal, period)