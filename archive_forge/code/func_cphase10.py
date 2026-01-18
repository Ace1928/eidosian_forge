from typing import Callable, cast, Dict, Union
import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
from cirq import Circuit, LineQubit
from cirq.ops import (
def cphase10(phi: float) -> TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE10 gate.

    In pyQuil, CPHASE10(phi) = diag(1, 1, [exp(1j * phi), 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [0, 0, phi, 0].

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE10 gate of given angle.
    """
    return TwoQubitDiagonalGate([0, 0, phi, 0])