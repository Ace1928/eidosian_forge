from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def lifted_gate_matrix(matrix: np.ndarray, qubit_inds: Sequence[int], n_qubits: int) -> np.ndarray:
    """
    Lift a unitary matrix to act on the specified qubits in a full ``n_qubits``-qubit
    Hilbert space.

    For 1-qubit gates, this is easy and can be achieved with appropriate kronning of identity
    matrices. For 2-qubit gates acting on adjacent qubit indices, it is also easy. However,
    for a multiqubit gate acting on non-adjactent qubit indices, we must first apply a permutation
    matrix to make the qubits adjacent and then apply the inverse permutation.

    :param matrix: A 2^k by 2^k matrix encoding an n-qubit operation, where ``k == len(qubit_inds)``
    :param qubit_inds: The qubit indices we wish the matrix to act on.
    :param n_qubits: The total number of qubits.
    :return: A 2^n by 2^n lifted version of the unitary matrix acting on the specified qubits.
    """
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols, 'Matrix must be square'
    gate_size = np.log2(n_rows)
    assert gate_size == int(gate_size), 'Matrix must be 2^n by 2^n'
    gate_size = int(gate_size)
    pi_permutation_matrix, final_map, start_i = permutation_arbitrary(qubit_inds, n_qubits)
    if start_i > 0:
        check = final_map[-gate_size - start_i:-start_i]
    else:
        check = final_map[-gate_size - start_i:]
    np.testing.assert_allclose(check, qubit_inds)
    v_matrix = qubit_adjacent_lifted_gate(start_i, matrix, n_qubits)
    return np.dot(np.conj(pi_permutation_matrix.T), np.dot(v_matrix, pi_permutation_matrix))