import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def quantile_ci(self, p, alpha=0.05, method='cloglog'):
    """
        Returns a confidence interval for a survival quantile.

        Parameters
        ----------
        p : float
            The probability point for which a confidence interval is
            determined.
        alpha : float
            The confidence interval has nominal coverage probability
            1 - `alpha`.
        method : str
            Function to use for g-transformation, must be ...

        Returns
        -------
        lb : float
            The lower confidence limit.
        ub : float
            The upper confidence limit.

        Notes
        -----
        The confidence interval is obtained by inverting Z-tests.  The
        limits of the confidence interval will always be observed
        event times.

        References
        ----------
        The method is based on the approach used in SAS, documented here:

          http://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/viewer.htm#statug_lifetest_details03.htm
        """
    tr = norm.ppf(1 - alpha / 2)
    method = method.lower()
    if method == 'cloglog':
        g = lambda x: np.log(-np.log(x))
        gprime = lambda x: -1 / (x * np.log(x))
    elif method == 'linear':
        g = lambda x: x
        gprime = lambda x: 1
    elif method == 'log':
        g = np.log
        gprime = lambda x: 1 / x
    elif method == 'logit':
        g = lambda x: np.log(x / (1 - x))
        gprime = lambda x: 1 / (x * (1 - x))
    elif method == 'asinsqrt':
        g = lambda x: np.arcsin(np.sqrt(x))
        gprime = lambda x: 1 / (2 * np.sqrt(x) * np.sqrt(1 - x))
    else:
        raise ValueError('unknown method')
    r = g(self.surv_prob) - g(1 - p)
    r /= gprime(self.surv_prob) * self.surv_prob_se
    ii = np.flatnonzero(np.abs(r) <= tr)
    if len(ii) == 0:
        return (np.nan, np.nan)
    lb = self.surv_times[ii[0]]
    if ii[-1] == len(self.surv_times) - 1:
        ub = np.inf
    else:
        ub = self.surv_times[ii[-1] + 1]
    return (lb, ub)