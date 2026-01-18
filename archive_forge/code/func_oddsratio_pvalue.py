import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def oddsratio_pvalue(self, null=1):
    """
        P-value for a hypothesis test about the odds ratio.

        Parameters
        ----------
        null : float
            The null value of the odds ratio.
        """
    return self.log_oddsratio_pvalue(np.log(null))