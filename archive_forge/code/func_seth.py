from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def seth(self, value):
    """Setter for kernel bandwidth, h"""
    self._h = value