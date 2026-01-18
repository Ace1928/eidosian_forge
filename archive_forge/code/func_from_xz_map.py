from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_xz_map(x_to: Tuple[Pauli, bool], z_to: Tuple[Pauli, bool]) -> 'SingleQubitCliffordGate':
    """Returns a SingleQubitCliffordGate for the specified transforms.
        The Y transform is derived from the X and Z.

        Args:
            x_to: Which Pauli to transform X to and if it should negate.
            z_to: Which Pauli to transform Z to and if it should negate.
        """
    return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(x_to=x_to, z_to=z_to))