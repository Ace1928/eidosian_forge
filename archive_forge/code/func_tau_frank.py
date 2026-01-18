import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def tau_frank(theta):
    """Kendall's tau for Frank Copula

    This uses Taylor series expansion for theta <= 1.

    Parameters
    ----------
    theta : float
        Parameter of the Frank copula. (not vectorized)

    Returns
    -------
    tau : float, tau for given theta
    """
    if theta <= 1:
        tau = _tau_frank_expansion(theta)
    else:
        debye_value = _debye(theta)
        tau = 1 + 4 * (debye_value - 1) / theta
    return tau