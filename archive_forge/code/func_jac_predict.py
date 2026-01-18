import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
def jac_predict(self, params):
    """jacobian of prediction function using complex step derivative

        This assumes that the predict function does not use complex variable
        but is designed to do so.

        """
    from statsmodels.tools.numdiff import approx_fprime_cs
    jaccs_err = approx_fprime_cs(params, self._predict)
    return jaccs_err