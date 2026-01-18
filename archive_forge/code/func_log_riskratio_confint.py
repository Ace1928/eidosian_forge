import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def log_riskratio_confint(self, alpha=0.05, method='normal'):
    """
        A confidence interval for the log risk ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
    f = -stats.norm.ppf(alpha / 2)
    lrr = self.log_riskratio
    se = self.log_riskratio_se
    lcb = lrr - f * se
    ucb = lrr + f * se
    return (lcb, ucb)