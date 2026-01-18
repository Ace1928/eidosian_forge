from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def density_confint(self, density, nobs, alpha=0.05):
    """approximate pointwise confidence interval for kernel density

        The confidence interval is centered at the estimated density and
        ignores the bias of the density estimate.

        not verified

        Parameters
        ----------
        density : array_lie
            pdf of the kernel density
        nobs : int
            number of observations used in the KDE estimation

        Returns
        -------
        conf_int : ndarray
            estimated confidence interval of the density estimate, lower bound
            in first column and upper bound in second column

        Notes
        -----
        This uses the asymptotic normal approximation to the distribution of
        the density estimate. The lower bound can be negative for density
        values close to zero.
        """
    from scipy import stats
    crit = stats.norm.isf(alpha / 2.0)
    density = np.asarray(density)
    half_width = crit * np.sqrt(self.density_var(density, nobs))
    conf_int = np.column_stack((density - half_width, density + half_width))
    return conf_int