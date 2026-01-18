import numpy as np
from ._penalties import NonePenalty
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime
def hessian_numdiff(self, params, pen_weight=None, **kwds):
    """hessian based on finite difference derivative
        """
    if pen_weight is None:
        pen_weight = self.pen_weight
    loglike = lambda p: self.loglike(p, pen_weight=pen_weight, **kwds)
    from statsmodels.tools.numdiff import approx_hess
    return approx_hess(params, loglike)