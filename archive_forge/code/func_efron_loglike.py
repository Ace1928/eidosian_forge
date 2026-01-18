import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
def efron_loglike(self, params):
    """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Efron method to handle tied
        times.
        """
    surv = self.surv
    like = 0.0
    for stx in range(surv.nstrat):
        exog_s = surv.exog_s[stx]
        linpred = np.dot(exog_s, params)
        if surv.offset_s is not None:
            linpred += surv.offset_s[stx]
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)
        xp0 = 0.0
        uft_ix = surv.ufailt_ix[stx]
        nuft = len(uft_ix)
        for i in range(nuft)[::-1]:
            ix = surv.risk_enter[stx][i]
            xp0 += e_linpred[ix].sum()
            xp0f = e_linpred[uft_ix[i]].sum()
            ix = uft_ix[i]
            like += linpred[ix].sum()
            m = len(ix)
            J = np.arange(m, dtype=np.float64) / m
            like -= np.log(xp0 - J * xp0f).sum()
            ix = surv.risk_exit[stx][i]
            xp0 -= e_linpred[ix].sum()
    return like