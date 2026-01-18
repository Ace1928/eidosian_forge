import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def mv_test_mean(self, mu_array, return_weights=False):
    """
        Returns -2 x log likelihood and the p-value
        for a multivariate hypothesis test of the mean

        Parameters
        ----------
        mu_array  : 1d array
            Hypothesized values for the mean.  Must have same number of
            elements as columns in endog

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of mu_array. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value for mu_array
        """
    endog = self.endog
    nobs = self.nobs
    if len(mu_array) != endog.shape[1]:
        raise ValueError('mu_array must have the same number of elements as the columns of the data.')
    mu_array = mu_array.reshape(1, endog.shape[1])
    means = np.ones((endog.shape[0], endog.shape[1]))
    means = mu_array * means
    est_vect = endog - means
    start_vals = 1.0 / nobs * np.ones(endog.shape[1])
    eta_star = self._modif_newton(start_vals, est_vect, np.ones(nobs) * (1.0 / nobs))
    denom = 1 + np.dot(eta_star, est_vect.T)
    self.new_weights = 1 / nobs * 1 / denom
    llr = -2 * np.sum(np.log(nobs * self.new_weights))
    p_val = chi2.sf(llr, mu_array.shape[1])
    if return_weights:
        return (llr, p_val, self.new_weights.T)
    else:
        return (llr, p_val)