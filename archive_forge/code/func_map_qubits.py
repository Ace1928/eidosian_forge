import abc
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq import protocols
from cirq.ops import pauli_string as ps, raw_types
@abc.abstractmethod
def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]) -> Self:
    """Return an equivalent operation on new qubits with its Pauli string
        mapped to new qubits.

        new_pauli_string = self.pauli_string.map_qubits(qubit_map)
        """