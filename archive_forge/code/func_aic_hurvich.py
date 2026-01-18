import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def aic_hurvich(self, bw, func=None):
    """
        Computes the AIC Hurvich criteria for the estimation of the bandwidth.

        Parameters
        ----------
        bw : str or array_like
            See the ``bw`` parameter of `KernelReg` for details.

        Returns
        -------
        aic : ndarray
            The AIC Hurvich criteria, one element for each variable.
        func : None
            Unused here, needed in signature because it's used in `cv_loo`.

        References
        ----------
        See ch.2 in [1] and p.35 in [2].
        """
    H = np.empty((self.nobs, self.nobs))
    for j in range(self.nobs):
        H[:, j] = gpke(bw, data=self.exog, data_predict=self.exog[j, :], ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, var_type=self.var_type, tosum=False)
    denom = H.sum(axis=1)
    H = H / denom
    gx = KernelReg(endog=self.endog, exog=self.exog, var_type=self.var_type, reg_type=self.reg_type, bw=bw, defaults=EstimatorSettings(efficient=False)).fit()[0]
    gx = np.reshape(gx, (self.nobs, 1))
    sigma = ((self.endog - gx) ** 2).sum(axis=0) / float(self.nobs)
    frac = (1 + np.trace(H) / float(self.nobs)) / (1 - (np.trace(H) + 2) / float(self.nobs))
    aic = np.log(sigma) + frac
    return aic