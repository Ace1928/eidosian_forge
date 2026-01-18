import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
def efron_hessian(self, params):
    """
        Returns the Hessian matrix of the partial log-likelihood
        evaluated at `params`, using the Efron method to handle tied
        times.
        """
    surv = self.surv
    hess = 0.0
    for stx in range(surv.nstrat):
        exog_s = surv.exog_s[stx]
        linpred = np.dot(exog_s, params)
        if surv.offset_s is not None:
            linpred += surv.offset_s[stx]
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)
        xp0, xp1, xp2 = (0.0, 0.0, 0.0)
        uft_ix = surv.ufailt_ix[stx]
        nuft = len(uft_ix)
        for i in range(nuft)[::-1]:
            ix = surv.risk_enter[stx][i]
            if len(ix) > 0:
                xp0 += e_linpred[ix].sum()
                v = exog_s[ix, :]
                xp1 += (e_linpred[ix][:, None] * v).sum(0)
                elx = e_linpred[ix]
                xp2 += np.einsum('ij,ik,i->jk', v, v, elx)
            ixf = uft_ix[i]
            if len(ixf) > 0:
                v = exog_s[ixf, :]
                xp0f = e_linpred[ixf].sum()
                xp1f = (e_linpred[ixf][:, None] * v).sum(0)
                elx = e_linpred[ixf]
                xp2f = np.einsum('ij,ik,i->jk', v, v, elx)
            m = len(uft_ix[i])
            J = np.arange(m, dtype=np.float64) / m
            c0 = xp0 - J * xp0f
            hess += xp2 * np.sum(1 / c0)
            hess -= xp2f * np.sum(J / c0)
            mat = (xp1[None, :] - np.outer(J, xp1f)) / c0[:, None]
            hess -= np.einsum('ij,ik->jk', mat, mat)
            ix = surv.risk_exit[stx][i]
            if len(ix) > 0:
                xp0 -= e_linpred[ix].sum()
                v = exog_s[ix, :]
                xp1 -= (e_linpred[ix][:, None] * v).sum(0)
                elx = e_linpred[ix]
                xp2 -= np.einsum('ij,ik,i->jk', v, v, elx)
    return -hess