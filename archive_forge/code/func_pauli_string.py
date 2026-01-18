import abc
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq import protocols
from cirq.ops import pauli_string as ps, raw_types
@property
def pauli_string(self) -> 'cirq.PauliString':
    return self._pauli_string