import inspect
import warnings
import numpy as np
from scipy import special, stats
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
from . import links as L, varfuncs as V
def starting_mu(self, y):
    """
        The starting values for the IRLS algorithm for the Binomial family.
        A good choice for the binomial family is :math:`\\mu_0 = (Y_i + 0.5)/2`
        """
    return (y + 0.5) / 2