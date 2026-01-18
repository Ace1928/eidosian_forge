import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def sum_weights(self):
    """Sum of weights"""
    return self.weights.sum(0)