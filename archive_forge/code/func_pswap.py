from typing import Callable, cast, Dict, Union
import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
from cirq import Circuit, LineQubit
from cirq.ops import (
def pswap(phi: float) -> MatrixGate:
    """Returns a Cirq MatrixGate for pyQuil's PSWAP gate.

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A MatrixGate equivalent to a PSWAP gate of given angle.
    """
    pswap_matrix = np.array([[1, 0, 0, 0], [0, 0, np.exp(1j * phi), 0], [0, np.exp(1j * phi), 0, 0], [0, 0, 0, 1]], dtype=complex)
    return MatrixGate(pswap_matrix)