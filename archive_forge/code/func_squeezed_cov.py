import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def squeezed_cov(r, phi, hbar=2):
    """Returns the squeezed covariance matrix of a squeezed state.

    Args:
        r (float): the squeezing magnitude
        p (float): the squeezing phase :math:`\\phi`
        hbar (float): (default 2) the value of :math:`\\hbar` in the commutation
            relation :math:`[\\x,\\p]=i\\hbar`
    Returns:
        array: the squeezed state
    """
    cov = np.array([[math.exp(-2 * r), 0], [0, math.exp(2 * r)]]) * hbar / 2
    R = rotation(phi / 2)
    return R @ cov @ R.T