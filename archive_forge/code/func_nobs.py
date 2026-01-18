import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def nobs(self):
    """alias for number of observations/cases, equal to sum of weights
        """
    return self.sum_weights