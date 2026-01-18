from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_single_map(pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]]=None, *, x_to: Optional[Tuple[Pauli, bool]]=None, y_to: Optional[Tuple[Pauli, bool]]=None, z_to: Optional[Tuple[Pauli, bool]]=None) -> 'SingleQubitCliffordGate':
    """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        The arguments are exclusive, only one may be specified.

        Args:
            pauli_map_to: A dictionary with a single key value pair describing
                the transform.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
    rotation_map = _validate_map_input(1, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
    (trans_from, (trans_to, flip)), = tuple(rotation_map.items())
    if trans_from == trans_to:
        trans_from2 = Pauli.by_relative_index(trans_to, 1)
        trans_to2 = Pauli.by_relative_index(trans_from, 1)
        flip2 = False
    else:
        trans_from2 = trans_to
        trans_to2 = trans_from
        flip2 = not flip
    rotation_map[trans_from2] = (trans_to2, flip2)
    return SingleQubitCliffordGate.from_double_map(rotation_map)