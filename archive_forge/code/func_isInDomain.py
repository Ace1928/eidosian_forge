from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def isInDomain(xy):
    """Used for filter to check if point is in the domain"""
    u = (xy[0] - x) / self.h
    return np.all((u >= self.domain[0]) & (u <= self.domain[1]))