import numpy as np
import scipy.stats
import warnings
def deriv2_numdiff(self, p):
    """
        Second derivative of the link function g''(p)

        implemented through numerical differentiation
        """
    from statsmodels.tools.numdiff import _approx_fprime_scalar
    p = np.atleast_1d(p)
    return _approx_fprime_scalar(p, self.deriv, centered=True)