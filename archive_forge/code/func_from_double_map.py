from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_double_map(pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]]=None, *, x_to: Optional[Tuple[Pauli, bool]]=None, y_to: Optional[Tuple[Pauli, bool]]=None, z_to: Optional[Tuple[Pauli, bool]]=None) -> 'SingleQubitCliffordGate':
    """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        Either pauli_map_to or two of (x_to, y_to, z_to) may be specified.

        Args:
            pauli_map_to: A dictionary with two key value pairs describing
                two transforms.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
    rotation_map = _validate_map_input(2, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
    (from1, trans1), (from2, trans2) = tuple(rotation_map.items())
    from3 = from1.third(from2)
    to3 = trans1[0].third(trans2[0])
    flip3 = trans1[1] ^ trans2[1] ^ ((from1 < from2) != (trans1[0] < trans2[0]))
    rotation_map[from3] = (to3, flip3)
    return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(rotation_map))