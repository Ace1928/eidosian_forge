from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def density_var(self, density, nobs):
    """approximate pointwise variance for kernel density

        not verified

        Parameters
        ----------
        density : array_lie
            pdf of the kernel density
        nobs : int
            number of observations used in the KDE estimation

        Returns
        -------
        kde_var : ndarray
            estimated variance of the density estimate

        Notes
        -----
        This uses the asymptotic normal approximation to the distribution of
        the density estimate.
        """
    return np.asarray(density) * self.L2Norm / self.h / nobs