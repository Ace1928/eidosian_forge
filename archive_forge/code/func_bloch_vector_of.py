import abc
from typing import List, Mapping, Optional, Tuple, TYPE_CHECKING, Sequence
import numpy as np
from cirq import linalg, qis, value
from cirq.sim import simulator, simulation_utils
def bloch_vector_of(self, qubit: 'cirq.Qid') -> np.ndarray:
    """Returns the bloch vector of a qubit in the state.

        Calculates the bloch vector of the given qubit
        in the state given by self.state_vector(), given that
        self.state_vector() follows the standard Kronecker convention of
        numpy.kron.

        Args:
            qubit: qubit who's bloch vector we want to find.

        Returns:
            A length 3 numpy array representing the qubit's bloch vector.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if index is out of range for the number of qubits
                corresponding to the state.
        """
    return qis.bloch_vector_from_state_vector(self.state_vector(), self.qubit_map[qubit], qid_shape=self._qid_shape)