import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def ci_mean(self, sig=0.05, method='gamma', epsilon=10 ** (-8), gamma_low=-10 ** 10, gamma_high=10 ** 10):
    """
        Returns the confidence interval for the mean.

        Parameters
        ----------
        sig : float
            significance level. Default is .05

        method : str
            Root finding method,  Can be 'nested-brent' or
            'gamma'.  Default is 'gamma'

            'gamma' Tries to solve for the gamma parameter in the
            Lagrange (see Owen pg 22) and then determine the weights.

            'nested brent' uses brents method to find the confidence
            intervals but must maximize the likelihood ratio on every
            iteration.

            gamma is generally much faster.  If the optimizations does not
            converge, try expanding the gamma_high and gamma_low
            variable.

        gamma_low : float
            Lower bound for gamma when finding lower limit.
            If function returns f(a) and f(b) must have different signs,
            consider lowering gamma_low.

        gamma_high : float
            Upper bound for gamma when finding upper limit.
            If function returns f(a) and f(b) must have different signs,
            consider raising gamma_high.

        epsilon : float
            When using 'nested-brent', amount to decrease (increase)
            from the maximum (minimum) of the data when
            starting the search.  This is to protect against the
            likelihood ratio being zero at the maximum (minimum)
            value of the data.  If data is very small in absolute value
            (<10 ``**`` -6) consider shrinking epsilon

            When using 'gamma', amount to decrease (increase) the
            minimum (maximum) by to start the search for gamma.
            If function returns f(a) and f(b) must have different signs,
            consider lowering epsilon.

        Returns
        -------
        Interval : tuple
            Confidence interval for the mean
        """
    endog = self.endog
    sig = 1 - sig
    if method == 'nested-brent':
        self.r0 = chi2.ppf(sig, 1)
        middle = np.mean(endog)
        epsilon_u = (max(endog) - np.mean(endog)) * epsilon
        epsilon_l = (np.mean(endog) - min(endog)) * epsilon
        ulim = optimize.brentq(self._ci_limits_mu, middle, max(endog) - epsilon_u)
        llim = optimize.brentq(self._ci_limits_mu, middle, min(endog) + epsilon_l)
        return (llim, ulim)
    if method == 'gamma':
        self.r0 = chi2.ppf(sig, 1)
        gamma_star_l = optimize.brentq(self._find_gamma, gamma_low, min(endog) - epsilon)
        gamma_star_u = optimize.brentq(self._find_gamma, max(endog) + epsilon, gamma_high)
        weights_low = (endog - gamma_star_l) ** (-1) / np.sum((endog - gamma_star_l) ** (-1))
        weights_high = (endog - gamma_star_u) ** (-1) / np.sum((endog - gamma_star_u) ** (-1))
        mu_low = np.sum(weights_low * endog)
        mu_high = np.sum(weights_high * endog)
        return (mu_low, mu_high)