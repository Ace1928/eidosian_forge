import math
import warnings
import numpy as np
import scipy.linalg
from ._optimize import (_check_unknown_options, _status_message,
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.optimize._differentiable_functions import FD_METHODS
@property
def jac_mag(self):
    """Magnitude of jacobian of objective function at current iteration."""
    if self._g_mag is None:
        self._g_mag = scipy.linalg.norm(self.jac)
    return self._g_mag