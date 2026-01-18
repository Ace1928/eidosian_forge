import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def ci_corr(self, sig=0.05, upper_bound=None, lower_bound=None):
    """
        Returns the confidence intervals for the correlation coefficient

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value the upper confidence limit can be.
            Default is  99% confidence limit assuming normality.

        lower_bound : float
            Minimum value the lower confidence limit can be.
            Default is 99% confidence limit assuming normality.

        Returns
        -------
        interval : tuple
            Confidence interval for the correlation
        """
    endog = self.endog
    nobs = self.nobs
    self.r0 = chi2.ppf(1 - sig, 1)
    point_est = np.corrcoef(endog[:, 0], endog[:, 1])[0, 1]
    if upper_bound is None:
        upper_bound = min(0.999, point_est + 2.5 * ((1.0 - point_est ** 2.0) / (nobs - 2.0)) ** 0.5)
    if lower_bound is None:
        lower_bound = max(-0.999, point_est - 2.5 * np.sqrt((1.0 - point_est ** 2.0) / (nobs - 2.0)))
    llim = optimize.brenth(self._ci_limits_corr, lower_bound, point_est)
    ulim = optimize.brenth(self._ci_limits_corr, point_est, upper_bound)
    return (llim, ulim)