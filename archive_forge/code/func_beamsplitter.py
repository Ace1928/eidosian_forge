import math
import cmath
import numpy as np
from scipy.special import factorial as fac
import pennylane as qml
from pennylane.ops import Identity
from pennylane import Device
from .._version import __version__
def beamsplitter(theta, phi):
    """Beamsplitter.

    Args:
        theta (float): transmittivity angle (:math:`t=\\cos\\theta`)
        phi (float): phase angle (:math:`r=e^{i\\phi}\\sin\\theta`)

    Returns:
        array: symplectic transformation matrix
    """
    cp = math.cos(phi)
    sp = math.sin(phi)
    ct = math.cos(theta)
    st = math.sin(theta)
    S = np.array([[ct, -cp * st, 0, -st * sp], [cp * st, ct, -st * sp, 0], [0, st * sp, ct, -cp * st], [st * sp, 0, cp * st, ct]])
    return S