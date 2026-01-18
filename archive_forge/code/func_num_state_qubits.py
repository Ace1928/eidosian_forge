from typing import Optional
from qiskit.circuit import QuantumCircuit
@property
def num_state_qubits(self) -> int:
    """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
    return self._num_state_qubits