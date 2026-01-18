import itertools
from typing import Dict, List, Tuple, cast
import numpy as np
from pyquil.paulis import PauliTerm
def euler_angles_RX(theta: float) -> Tuple[float, float, float]:
    """
    A tuple of angles which corresponds to a ZXZXZ-decomposed ``RX`` gate.

    :param theta: The angle parameter for the ``RX`` gate.
    :return: The corresponding Euler angles for that gate.
    """
    return (np.pi / 2, theta, -np.pi / 2)