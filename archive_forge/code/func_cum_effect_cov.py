import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def cum_effect_cov(self, orth=False):
    """
        Compute asymptotic standard errors for cumulative impulse response
        coefficients

        Parameters
        ----------
        orth : bool

        Notes
        -----
        eq. 3.7.7 (non-orth), 3.7.10 (orth)

        Returns
        -------
        """
    Ik = np.eye(self.neqs)
    PIk = np.kron(self.P.T, Ik)
    F = 0.0
    covs = self._empty_covm(self.periods + 1)
    for i in range(self.periods + 1):
        if i > 0:
            F = F + self.G[i - 1]
        if orth:
            if i == 0:
                apiece = 0
            else:
                Bn = np.dot(PIk, F)
                apiece = Bn @ self.cov_a @ Bn.T
            Bnbar = np.dot(np.kron(Ik, self.cum_effects[i]), self.H)
            bpiece = Bnbar @ self.cov_sig @ Bnbar.T / self.T
            covs[i] = apiece + bpiece
        else:
            if i == 0:
                covs[i] = np.zeros((self.neqs ** 2, self.neqs ** 2))
                continue
            covs[i] = F @ self.cov_a @ F.T
    return covs