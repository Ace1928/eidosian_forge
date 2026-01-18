import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def test_kurt(self, kurt0, return_weights=False):
    """
        Returns -2 x log-likelihood and the p-value for the hypothesized
        kurtosis.

        Parameters
        ----------
        kurt0 : float
            Kurtosis value to be tested

        return_weights : bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value of kurt0
        """
    self.kurt0 = kurt0
    start_nuisance = np.array([self.endog.mean(), self.endog.var()])
    llr = optimize.fmin_powell(self._opt_kurt, start_nuisance, full_output=1, disp=0)[1]
    p_val = chi2.sf(llr, 1)
    if return_weights:
        return (llr, p_val, self.new_weights.T)
    return (llr, p_val)