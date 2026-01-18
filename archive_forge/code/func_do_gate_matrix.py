from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> 'NumpyWavefunctionSimulator':
    """
        Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """
    tensor = np.reshape(matrix, (2,) * len(qubits) * 2)
    self.wf = targeted_tensordot(gate=tensor, wf=self.wf, wf_target_inds=qubits)
    return self