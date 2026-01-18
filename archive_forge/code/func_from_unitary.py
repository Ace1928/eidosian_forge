from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_unitary(u: np.ndarray) -> Optional['SingleQubitCliffordGate']:
    """Creates Clifford gate with given unitary (up to global phase).

        Args:
            u: 2x2 unitary matrix of a Clifford gate.

        Returns:
            SingleQubitCliffordGate, whose matrix is equal to given matrix (up
            to global phase), or `None` if `u` is not a matrix of a single-qubit
            Clifford gate.
        """
    if u.shape != (2, 2) or not linalg.is_unitary(u):
        return None
    x = protocols.unitary(pauli_gates.X)
    z = protocols.unitary(pauli_gates.Z)
    x_to = _to_pauli_tuple(u @ x @ u.conj().T)
    z_to = _to_pauli_tuple(u @ z @ u.conj().T)
    if x_to is None or z_to is None:
        return None
    return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(x_to=x_to, z_to=z_to))