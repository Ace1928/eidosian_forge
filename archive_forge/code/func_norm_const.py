from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
@property
def norm_const(self):
    """
        Normalising constant for kernel (integral from -inf to inf)
        """
    if self._normconst is None:
        if self.domain is None:
            quadres = scipy.integrate.quad(self._shape, -inf, inf)
        else:
            quadres = scipy.integrate.quad(self._shape, self.domain[0], self.domain[1])
        self._normconst = 1.0 / quadres[0]
    return self._normconst