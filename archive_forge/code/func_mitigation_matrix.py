from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag
def mitigation_matrix(self, qubits: Optional[Union[List[int], int]]=None) -> np.ndarray:
    """Return the measurement mitigation matrix for the specified qubits.

        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.

        Args:
            qubits: Optional, qubits being measured for operator expval.
                    if a single int is given, it is assumed to be the index
                    of the qubit in self._qubits

        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """
    if qubits is None:
        qubits = self._qubits
    if isinstance(qubits, int):
        qubits = [self._qubits[qubits]]
    qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
    mat = self._mitigation_mats[qubit_indices[0]]
    for i in qubit_indices[1:]:
        mat = np.kron(self._mitigation_mats[i], mat)
    return mat