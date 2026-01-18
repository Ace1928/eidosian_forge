import math
import warnings
import numpy as np
import scipy.linalg
from ._optimize import (_check_unknown_options, _status_message,
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.optimize._differentiable_functions import FD_METHODS
def get_boundaries_intersections(self, z, d, trust_radius):
    """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius ** 2
    sqrt_discriminant = math.sqrt(b * b - 4 * a * c)
    aux = b + math.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return sorted([ta, tb])