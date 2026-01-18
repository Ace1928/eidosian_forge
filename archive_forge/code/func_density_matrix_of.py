import abc
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING, Sequence
import numpy as np
from cirq import linalg, qis, value
from cirq.sim import simulator, simulation_utils
def density_matrix_of(self, qubits: Optional[List['cirq.Qid']]=None) -> np.ndarray:
    """Returns the density matrix of the state.

        Calculate the density matrix for the system on the qubits provided.
        Any qubits not in the list that are present in self.state_vector() will
        be traced out. If qubits is None, the full density matrix for
        self.state_vector() is returned, given self.state_vector() follows
        standard Kronecker convention of numpy.kron.

        For example, if `self.state_vector()` returns
        `np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)`,
        then `density_matrix_of(qubits = None)` gives us

        $$
        \\rho = \\begin{bmatrix}
                    0.5 & 0.5 \\\\
                    0.5 & 0.5
                \\end{bmatrix}
        $$

        Args:
            qubits: list containing qubit IDs that you would like
                to include in the density matrix (i.e.) qubits that WON'T
                be traced out.

        Returns:
            A numpy array representing the density matrix.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if the indices are out of range for the number of qubits
                corresponding to the state.
        """
    return qis.density_matrix_from_state_vector(self.state_vector(), [self.qubit_map[q] for q in qubits] if qubits is not None else None, qid_shape=self._qid_shape)