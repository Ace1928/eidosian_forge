import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def log_oddsratio_pvalue(self, null=0):
    """
        P-value for a hypothesis test about the log odds ratio.

        Parameters
        ----------
        null : float
            The null value of the log odds ratio.
        """
    zscore = (self.log_oddsratio - null) / self.log_oddsratio_se
    pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
    return pvalue