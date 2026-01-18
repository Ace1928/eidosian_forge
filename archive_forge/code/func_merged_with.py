from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def merged_with(self, second: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
    """Returns a SingleQubitCliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent up to global phase."""
    return SingleQubitCliffordGate.from_clifford_tableau(self.clifford_tableau.then(second.clifford_tableau))