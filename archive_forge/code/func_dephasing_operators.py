import cmath
from typing import Tuple
import numpy as np
def dephasing_operators(p: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the phase damping Kraus operators
    """
    k0 = np.eye(2) * np.sqrt(1 - p / 2)
    k1 = np.sqrt(p / 2) * Z
    return (k0, k1)