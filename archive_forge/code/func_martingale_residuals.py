import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def martingale_residuals(self):
    """
        The martingale residuals.
        """
    surv = self.model.surv
    mart_resid = np.nan * np.ones(len(self.model.endog), dtype=np.float64)
    cumhaz_f_list = self.baseline_cumulative_hazard_function
    for stx in range(surv.nstrat):
        cumhaz_f = cumhaz_f_list[stx]
        exog_s = surv.exog_s[stx]
        time_s = surv.time_s[stx]
        linpred = np.dot(exog_s, self.params)
        if surv.offset_s is not None:
            linpred += surv.offset_s[stx]
        e_linpred = np.exp(linpred)
        ii = surv.stratum_rows[stx]
        chaz = cumhaz_f(time_s)
        mart_resid[ii] = self.model.status[ii] - e_linpred * chaz
    return mart_resid