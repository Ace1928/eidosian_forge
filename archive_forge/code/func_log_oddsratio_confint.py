import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def log_oddsratio_confint(self, alpha=0.05, method='normal'):
    """
        A confidence level for the log odds ratio.

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
    lor = self.log_oddsratio
    se = self.log_oddsratio_se
    lcb = lor - f * se
    ucb = lor + f * se
    return (lcb, ucb)