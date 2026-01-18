import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula
def spearmans_rho(self, corr=None):
    """
        Bivariate Spearman's rho based on correlation coefficient.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Spearman's rho that corresponds to pearson correlation in the
        elliptical copula.
        """
    if corr is None:
        corr = self.corr
    if corr.shape == (2, 2):
        corr = corr[0, 1]
    tau = 6 * np.arcsin(corr / 2) / np.pi
    return tau