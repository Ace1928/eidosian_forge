import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def standardized_resids(self):
    """
        Returns standardized residuals under independence.
        """
    row, col = self.marginal_probabilities
    sresids = self.resid_pearson / np.sqrt(np.outer(1 - row, 1 - col))
    return sresids