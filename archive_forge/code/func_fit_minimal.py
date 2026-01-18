import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
def fit_minimal(self, start_value, **kwargs):
    """minimal fitting with no extra calculations"""
    func = self.geterrors
    res = optimize.leastsq(func, start_value, full_output=0, **kwargs)
    return res