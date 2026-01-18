import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
def breslow_gradient(self, params):
    """
        Returns the gradient of the log partial likelihood, using the
        Breslow method to handle tied times.
        """
    surv = self.surv
    grad = 0.0
    for stx in range(surv.nstrat):
        strat_ix = surv.stratum_rows[stx]
        uft_ix = surv.ufailt_ix[stx]
        nuft = len(uft_ix)
        exog_s = surv.exog_s[stx]
        linpred = np.dot(exog_s, params)
        if surv.offset_s is not None:
            linpred += surv.offset_s[stx]
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)
        xp0, xp1 = (0.0, 0.0)
        for i in range(nuft)[::-1]:
            ix = surv.risk_enter[stx][i]
            if len(ix) > 0:
                v = exog_s[ix, :]
                xp0 += e_linpred[ix].sum()
                xp1 += (e_linpred[ix][:, None] * v).sum(0)
            ix = uft_ix[i]
            grad += (exog_s[ix, :] - xp1 / xp0).sum(0)
            ix = surv.risk_exit[stx][i]
            if len(ix) > 0:
                v = exog_s[ix, :]
                xp0 -= e_linpred[ix].sum()
                xp1 -= (e_linpred[ix][:, None] * v).sum(0)
    return grad