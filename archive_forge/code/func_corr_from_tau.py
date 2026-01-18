import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula
def corr_from_tau(self, tau):
    """Pearson correlation from kendall's tau.

        Parameters
        ----------
        tau : array_like
            Kendall's tau correlation coefficient.

        Returns
        -------
        Pearson correlation coefficient for given tau in elliptical
        copula. This can be used as parameter for an elliptical copula.
        """
    corr = np.sin(tau * np.pi / 2)
    return corr