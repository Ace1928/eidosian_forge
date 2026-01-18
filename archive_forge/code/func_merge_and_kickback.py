import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def merge_and_kickback(qubit: TKey, pauli_left: Optional[pauli_gates.Pauli], pauli_right: Optional[pauli_gates.Pauli], inv: bool) -> int:
    assert pauli_left is not None or pauli_right is not None
    if pauli_left is None or pauli_right is None:
        pauli_map[qubit] = cast(pauli_gates.Pauli, pauli_left or pauli_right)
        return 0
    if pauli_left == pauli_right:
        del pauli_map[qubit]
        return 0
    pauli_map[qubit] = pauli_left.third(pauli_right)
    if (pauli_left < pauli_right) ^ after_to_before:
        return int(inv) * 2 + 1
    return int(inv) * 2 - 1