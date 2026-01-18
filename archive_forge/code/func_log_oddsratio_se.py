import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def log_oddsratio_se(self):
    """
        Returns the standard error for the log odds ratio.
        """
    return np.sqrt(np.sum(1 / self.table))