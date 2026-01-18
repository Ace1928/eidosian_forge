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
def bounds_test(self, case: Literal[1, 2, 3, 4, 5], cov_type: str='nonrobust', cov_kwds: dict[str, Any]=None, use_t: bool=True, asymptotic: bool=True, nsim: int=100000, seed: int | Sequence[int] | np.random.RandomState | np.random.Generator | None=None):
    """
        Cointegration bounds test of Pesaran, Shin, and Smith

        Parameters
        ----------
        case : {1, 2, 3, 4, 5}
            One of the cases covered in the PSS test.
        cov_type : str
            The covariance estimator to use. The asymptotic distribution of
            the PSS test has only been established in the homoskedastic case,
            which is the default.

            The most common choices are listed below.  Supports all covariance
            estimators that are available in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlags` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that small-sample corrections should be applied
            to the covariance estimator.
        asymptotic : bool
            Flag indicating whether to use asymptotic critical values which
            were computed by simulation (True, default) or to simulate a
            sample-size specific set of critical values. Tables are only
            available for up to 10 components in the cointegrating
            relationship, so if more variables are included then simulation
            is always used. The simulation computed the test statistic under
            and assumption that the residuals are homoskedastic.
        nsim : int
            Number of simulations to run when computing exact critical values.
            Only used if ``asymptotic`` is ``True``.
        seed : {None, int, sequence[int], RandomState, Generator}, optional
            Seed to use when simulating critical values. Must be provided if
            reproducible critical value and p-values are required when
            ``asymptotic`` is ``False``.

        Returns
        -------
        BoundsTestResult
            Named tuple containing ``stat``, ``crit_vals``, ``p_values``,
            ``null` and ``alternative``. The statistic is the F-type
            test statistic favored in PSS.

        Notes
        -----
        The PSS bounds test has 5 cases which test the coefficients on the
        level terms in the model

        .. math::

           \\Delta Y_{t}=\\delta_{0} + \\delta_{1}t + Z_{t-1}\\beta
                        + \\sum_{j=0}^{P}\\Delta X_{t-j}\\Gamma + \\epsilon_{t}

        where :math:`Z_{t-1}` contains both :math:`Y_{t-1}` and
        :math:`X_{t-1}`.

        The cases determine which deterministic terms are included in the
        model and which are tested as part of the test.

        Cases:

        1. No deterministic terms
        2. Constant included in both the model and the test
        3. Constant included in the model but not in the test
        4. Constant and trend included in the model, only trend included in
           the test
        5. Constant and trend included in the model, neither included in the
           test

        The test statistic is a Wald-type quadratic form test that all of the
        coefficients in :math:`\\beta` are 0 along with any included
        deterministic terms, which depends on the case. The statistic returned
        is an F-type test statistic which is the standard quadratic form test
        statistic divided by the number of restrictions.

        References
        ----------
        .. [*] Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing
           approaches to the analysis of level relationships. Journal of
           applied econometrics, 16(3), 289-326.
        """
    model = self.model
    trend: Literal['n', 'c', 'ct']
    if case == 1:
        trend = 'n'
    elif case in (2, 3):
        trend = 'c'
    else:
        trend = 'ct'
    order = {key: max(val) for key, val in model._order.items()}
    uecm = UECM(model.data.endog, max(model.ar_lags), model.data.orig_exog, order=order, causal=model.causal, trend=trend)
    res = uecm.fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    cov = res.cov_params()
    nvar = len(res.model.ardl_order)
    if case == 1:
        rest = np.arange(nvar)
    elif case == 2:
        rest = np.arange(nvar + 1)
    elif case == 3:
        rest = np.arange(1, nvar + 1)
    elif case == 4:
        rest = np.arange(1, nvar + 2)
    elif case == 5:
        rest = np.arange(2, nvar + 2)
    r = np.zeros((rest.shape[0], cov.shape[1]))
    for i, loc in enumerate(rest):
        r[i, loc] = 1
    vcv = r @ cov @ r.T
    coef = r @ res.params
    stat = coef.T @ np.linalg.inv(vcv) @ coef / r.shape[0]
    k = nvar
    if asymptotic and k <= 10:
        cv = pss_critical_values.crit_vals
        key = (k, case)
        upper = cv[key + (True,)]
        lower = cv[key + (False,)]
        crit_vals = pd.DataFrame({'lower': lower, 'upper': upper}, index=pss_critical_values.crit_percentiles)
        crit_vals.index.name = 'percentile'
        p_values = pd.Series({'lower': _pss_pvalue(stat, k, case, False), 'upper': _pss_pvalue(stat, k, case, True)})
    else:
        nobs = res.resid.shape[0]
        crit_vals, p_values = _pss_simulate(stat, k, case, nobs=nobs, nsim=nsim, seed=seed)
    return BoundsTestResult(stat, crit_vals, p_values, 'No Cointegration', 'Possible Cointegration')