import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def ci_kurt(self, sig=0.05, upper_bound=None, lower_bound=None):
    """
        Returns the confidence interval for kurtosis.

        Parameters
        ----------

        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of kurtosis the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of kurtosis the lower limit can be.
            Default is .99 confidence limit assuming normality.

        Returns
        -------
        Interval : tuple
            Lower and upper confidence limit

        Notes
        -----
        For small n, upper_bound and lower_bound may have to be
        provided by the user.  Consider using test_kurt to find
        values close to the desired significance level.

        If function returns f(a) and f(b) must have different signs, consider
        expanding the bounds.
        """
    endog = self.endog
    nobs = self.nobs
    if upper_bound is None:
        upper_bound = kurtosis(endog) + 2.5 * (2.0 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5) * ((nobs ** 2.0 - 1.0) / ((nobs - 3.0) * (nobs + 5.0))) ** 0.5
    if lower_bound is None:
        lower_bound = kurtosis(endog) - 2.5 * (2.0 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5) * ((nobs ** 2.0 - 1.0) / ((nobs - 3.0) * (nobs + 5.0))) ** 0.5
    self.r0 = chi2.ppf(1 - sig, 1)
    llim = optimize.brentq(self._ci_limits_kurt, lower_bound, kurtosis(endog))
    ulim = optimize.brentq(self._ci_limits_kurt, kurtosis(endog), upper_bound)
    return (llim, ulim)