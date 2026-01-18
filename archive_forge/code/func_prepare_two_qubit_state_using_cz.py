from typing import List, TYPE_CHECKING
import numpy as np
from cirq import ops, qis, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def prepare_two_qubit_state_using_cz(q0: 'cirq.Qid', q1: 'cirq.Qid', state: 'cirq.STATE_VECTOR_LIKE') -> List['cirq.Operation']:
    """Prepares the given 2q state from |00> using at-most 1 CZ gate + single qubit rotations.

    Entangled states are prepared using exactly 1 CZ gate while product states are prepared
    using only single qubit rotations (0 CZ gates)

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.

    Returns:
        List of operations (at-most 1 CZ + single qubit rotations) preparing `state` from |00>.
    """
    state_vector = qis.to_valid_state_vector(state, num_qubits=2)
    state_vector = state_vector / np.linalg.norm(state_vector)
    u, s, vh = np.linalg.svd(state_vector.reshape(2, 2))
    if np.isclose(s[0], 1):
        return _1q_matrices_to_ops(u, vh.T, q0, q1, True)
    alpha = np.arccos(np.clip(s[0], 0, 1))
    op_list = [ops.ry(2 * alpha).on(q0), ops.H.on(q1), ops.CZ.on(q0, q1)]
    intermediate_state = circuits.Circuit(op_list).final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64)
    u_CZ, _, vh_CZ = np.linalg.svd(intermediate_state.reshape(2, 2))
    return op_list + _1q_matrices_to_ops(np.dot(u, np.linalg.inv(u_CZ)), np.dot(vh.T, np.linalg.inv(vh_CZ.T)), q0, q1)