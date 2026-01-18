import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
def errorsumsquares(self, params):
    return (self.geterrors(params) ** 2).sum()