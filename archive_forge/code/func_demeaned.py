import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def demeaned(self):
    """data with weighted mean subtracted"""
    return self.data - self.mean