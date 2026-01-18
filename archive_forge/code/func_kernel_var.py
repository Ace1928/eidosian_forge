from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
@property
def kernel_var(self):
    """Returns the second moment of the kernel"""
    if self._kernel_var is None:
        func = lambda x: x ** 2 * self.norm_const * self._shape(x)
        if self.domain is None:
            self._kernel_var = scipy.integrate.quad(func, -inf, inf)[0]
        else:
            self._kernel_var = scipy.integrate.quad(func, self.domain[0], self.domain[1])[0]
    return self._kernel_var