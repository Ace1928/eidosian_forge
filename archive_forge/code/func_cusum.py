import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
@cache_readonly
def cusum(self):
    """
        Cumulative sum of standardized recursive residuals statistics

        Returns
        -------
        cusum : array_like
            An array of length `nobs - k_exog` holding the
            CUSUM statistics.

        Notes
        -----
        The CUSUM statistic takes the form:

        .. math::

            W_t = \\frac{1}{\\hat \\sigma} \\sum_{j=k+1}^t w_j

        where :math:`w_j` is the recursive residual at time :math:`j` and
        :math:`\\hat \\sigma` is the estimate of the standard deviation
        from the full sample.

        Excludes the first `k_exog` datapoints.

        Due to differences in the way :math:`\\hat \\sigma` is calculated, the
        output of this function differs slightly from the output in the
        R package strucchange and the Stata contributed .ado file cusum6. The
        calculation in this package is consistent with the description of
        Brown et al. (1975)

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
    d = max(self.nobs_diffuse, self.loglikelihood_burn)
    return np.cumsum(self.resid_recursive[d:]) / np.std(self.resid_recursive[d:], ddof=1)