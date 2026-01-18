from typing import Callable, cast, Dict, Union
import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
from cirq import Circuit, LineQubit
from cirq.ops import (
def cphase00(phi: float) -> TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE00 gate.

    In pyQuil, CPHASE00(phi) = diag([exp(1j * phi), 1, 1, 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [phi, 0, 0, 0].

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE00 gate of given angle.
    """
    return TwoQubitDiagonalGate([phi, 0, 0, 0])