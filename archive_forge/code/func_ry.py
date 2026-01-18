from typing import (
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import controlled_gate, eigen_gate, gate_features, raw_types, control_values as cv
from cirq.type_workarounds import NotImplementedType
from cirq.ops.swap_gates import ISWAP, SWAP, ISwapPowGate, SwapPowGate
from cirq.ops.measurement_gate import MeasurementGate
imports.
def ry(rads: value.TParamVal) -> Ry:
    """Returns a gate with the matrix $e^{-i Y t / 2}$ where $t=rads$."""
    return Ry(rads=rads)