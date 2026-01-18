from typing import Any, Dict, List, Sequence, TYPE_CHECKING, Tuple
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import gate_features, common_gates, eigen_gate, pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
@property
def invert1(self) -> bool:
    return self._invert1