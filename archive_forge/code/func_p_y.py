import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
@property
def p_y(self) -> float:
    """The probability that a Pauli Y and no other gate occurs."""
    if self._num_qubits != 1:
        raise ValueError('num_qubits should be 1')
    return self._error_probabilities.get('Y', 0.0)