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
def inplace_right_multiply_by(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.MutablePauliString':
    """Right-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to right-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was mutated.

        Raises:
            TypeError: `other` was not a `cirq.PAULI_STRING_LIKE`. `self`
                was not mutated.
        """
    if self._imul_helper_checkpoint(other, +1) is NotImplemented:
        raise TypeError(f'{other!r} is not cirq.PAULI_STRING_LIKE.')
    return self