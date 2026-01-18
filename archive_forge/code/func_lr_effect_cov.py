import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def lr_effect_cov(self, orth=False):
    """
        Returns
        -------
        """
    lre = self.lr_effects
    Finfty = np.kron(np.tile(lre.T, self.lags), lre)
    Ik = np.eye(self.neqs)
    if orth:
        Binf = np.dot(np.kron(self.P.T, np.eye(self.neqs)), Finfty)
        Binfbar = np.dot(np.kron(Ik, lre), self.H)
        return Binf @ self.cov_a @ Binf.T + Binfbar @ self.cov_sig @ Binfbar.T
    else:
        return Finfty @ self.cov_a @ Finfty.T