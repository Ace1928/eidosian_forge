import cmath
from typing import Tuple
import numpy as np
def relaxation_operators(p: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the amplitude damping Kraus operators
    """
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - p)]])
    k1 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]])
    return (k0, k1)