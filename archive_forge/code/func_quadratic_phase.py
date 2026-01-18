import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def quadratic_phase(s):
    """Quadratic phase shift.

    Args:
        s (float): gate parameter

    Returns:
        array: symplectic transformation matrix
    """
    return np.array([[1, 0], [s, 1]])