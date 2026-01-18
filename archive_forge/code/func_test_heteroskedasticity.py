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
def test_heteroskedasticity(self, method, alternative='two-sided', use_f=True):
    """
        Test for heteroskedasticity of standardized residuals

        Tests whether the sum-of-squares in the first third of the sample is
        significantly different than the sum-of-squares in the last third
        of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
        is of no heteroskedasticity.

        Parameters
        ----------
        method : {'breakvar', None}
            The statistical test for heteroskedasticity. Must be 'breakvar'
            for test of a break in the variance. If None, an attempt is
            made to select an appropriate test.
        alternative : str, 'increasing', 'decreasing' or 'two-sided'
            This specifies the alternative for the p-value calculation. Default
            is two-sided.
        use_f : bool, optional
            Whether or not to compare against the asymptotic distribution
            (chi-squared) or the approximate small-sample distribution (F).
            Default is True (i.e. default is to compare against an F
            distribution).

        Returns
        -------
        output : ndarray
            An array with `(test_statistic, pvalue)` for each endogenous
            variable. The array is then sized `(k_endog, 2)`. If the method is
            called as `het = res.test_heteroskedasticity()`, then `het[0]` is
            an array of size 2 corresponding to the first endogenous variable,
            where `het[0][0]` is the test statistic, and `het[0][1]` is the
            p-value.

        See Also
        --------
        statsmodels.tsa.stattools.breakvar_heteroskedasticity_test

        Notes
        -----
        The null hypothesis is of no heteroskedasticity.

        For :math:`h = [T/3]`, the test statistic is:

        .. math::

            H(h) = \\sum_{t=T-h+1}^T  \\tilde v_t^2
            \\Bigg / \\sum_{t=d+1}^{d+1+h} \\tilde v_t^2

        where :math:`d` = max(loglikelihood_burn, nobs_diffuse)` (usually
        corresponding to diffuse initialization under either the approximate
        or exact approach).

        This statistic can be tested against an :math:`F(h,h)` distribution.
        Alternatively, :math:`h H(h)` is asymptotically distributed according
        to :math:`\\chi_h^2`; this second test can be applied by passing
        `use_f=True` as an argument.

        See section 5.4 of [1]_ for the above formula and discussion, as well
        as additional details.

        TODO

        - Allow specification of :math:`h`

        References
        ----------
        .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
               *Models and the Kalman Filter.* Cambridge University Press.
        """
    if method is None:
        method = 'breakvar'
    if self.standardized_forecasts_error is None:
        raise ValueError('Cannot compute test statistic when standardized forecast errors have not been computed.')
    if method == 'breakvar':
        from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
        resid = self.filter_results.standardized_forecasts_error
        d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
        nobs_effective = self.nobs - d
        h = int(np.round(nobs_effective / 3))
        test_statistics = []
        p_values = []
        for i in range(self.model.k_endog):
            test_statistic, p_value = breakvar_heteroskedasticity_test(resid[i, d:], subset_length=h, alternative=alternative, use_f=use_f)
            test_statistics.append(test_statistic)
            p_values.append(p_value)
        output = np.c_[test_statistics, p_values]
    else:
        raise NotImplementedError('Invalid heteroskedasticity test method.')
    return output