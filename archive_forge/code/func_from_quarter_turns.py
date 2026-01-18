from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
@staticmethod
def from_quarter_turns(pauli: Pauli, quarter_turns: int) -> 'SingleQubitCliffordGate':
    quarter_turns = quarter_turns % 4
    if quarter_turns == 0:
        return SingleQubitCliffordGate.I
    if quarter_turns == 1:
        return SingleQubitCliffordGate.from_pauli(pauli, True)
    if quarter_turns == 2:
        return SingleQubitCliffordGate.from_pauli(pauli)
    return SingleQubitCliffordGate.from_pauli(pauli, True) ** (-1)