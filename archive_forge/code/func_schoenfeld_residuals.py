import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def schoenfeld_residuals(self):
    """
        A matrix containing the Schoenfeld residuals.

        Notes
        -----
        Schoenfeld residuals for censored observations are set to zero.
        """
    surv = self.model.surv
    w_avg = self.weighted_covariate_averages
    sch_resid = np.nan * np.ones(self.model.exog.shape, dtype=np.float64)
    for stx in range(surv.nstrat):
        uft = surv.ufailt[stx]
        exog_s = surv.exog_s[stx]
        time_s = surv.time_s[stx]
        strat_ix = surv.stratum_rows[stx]
        ii = np.searchsorted(uft, time_s)
        jj = np.flatnonzero(ii < len(uft))
        sch_resid[strat_ix[jj], :] = exog_s[jj, :] - w_avg[stx][ii[jj], :]
    jj = np.flatnonzero(self.model.status == 0)
    sch_resid[jj, :] = np.nan
    return sch_resid