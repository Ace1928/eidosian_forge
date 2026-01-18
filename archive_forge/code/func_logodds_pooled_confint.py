import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def logodds_pooled_confint(self, alpha=0.05, method='normal'):
    """
        A confidence interval for the pooled log odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.

        Returns
        -------
        lcb : float
            The lower confidence limit.
        ucb : float
            The upper confidence limit.
        """
    lor = np.log(self.oddsratio_pooled)
    lor_se = self.logodds_pooled_se
    f = -stats.norm.ppf(alpha / 2)
    lcb = lor - f * lor_se
    ucb = lor + f * lor_se
    return (lcb, ucb)