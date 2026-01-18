from typing import Any, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import all_bitstrings
def targeted_einsum(gate: np.ndarray, wf: np.ndarray, wf_target_inds: List[int]) -> np.ndarray:
    """Left-multiplies the given axes of the wf tensor by the given gate matrix.

    Note that the matrix must have a compatible tensor structure.
    For example, if you have an 6-qubit state vector ``wf`` with shape
    (2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation ``op`` with shape
    (2, 2, 2, 2), and you want to apply ``op`` to the 5th and 3rd qubits
    within ``input_state``, then the output state vector is computed as follows::

        output_state = targeted_einsum(op, input_state, [5, 3])

    This method also works when the right hand side is a matrix instead of a
    vector. If a unitary circuit's matrix is ``old_effect``, and you append
    a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
    at offset 1 and the target q4 is the qubit at offset 4, then the appended
    circuit's unitary matrix is computed as follows::

        new_effect = targeted_left_multiply(CNOT.reshape((2, 2, 2, 2)), old_effect, [1, 4])

    :param gate: What to left-multiply the target tensor by.
    :param wf: A tensor to carefully broadcast a left-multiply over.
    :param wf_target_inds: Which axes of the target are being operated on.
    :returns: The output tensor.
    """
    k = len(wf_target_inds)
    d = len(wf.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple((data_indices[q] for q in wf_target_inds))
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, wf_target_inds):
        output_indices[t] = w
    return np.einsum(gate, input_indices, wf, data_indices, output_indices)