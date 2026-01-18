from typing import Optional, Tuple, TYPE_CHECKING, List
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import common_gates, gate_features, eigen_gate
def riswap(rads: value.TParamVal) -> ISwapPowGate:
    """Returns gate with matrix exp(+i angle_rads (X⊗X + Y⊗Y) / 2)."""
    pi = sympy.pi if protocols.is_parameterized(rads) else np.pi
    return ISwapPowGate() ** (2 * rads / pi)