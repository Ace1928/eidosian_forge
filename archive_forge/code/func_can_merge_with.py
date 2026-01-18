from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
def can_merge_with(self, op: 'PauliStringPhasor') -> bool:
    """Checks whether the underlying PauliStrings can be merged."""
    return self.pauli_string.equal_up_to_coefficient(op.pauli_string) and self.qubits == op.qubits