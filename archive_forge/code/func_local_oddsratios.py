import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def local_oddsratios(self):
    """
        Returns local odds ratios.

        See documentation for local_log_oddsratios.
        """
    return np.exp(self.local_log_oddsratios)