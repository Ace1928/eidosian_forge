import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
def stabilizers(self) -> List['cirq.DensePauliString']:
    """Returns the stabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi>"""
    return [self._row_to_dense_pauli(i) for i in range(self.n, 2 * self.n)]