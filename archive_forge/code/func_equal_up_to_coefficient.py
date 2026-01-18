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
def equal_up_to_coefficient(self, other: 'cirq.PauliString') -> bool:
    """Returns true of `self` and `other` are equal pauli strings, ignoring the coefficient."""
    return self._qubit_pauli_map == other._qubit_pauli_map