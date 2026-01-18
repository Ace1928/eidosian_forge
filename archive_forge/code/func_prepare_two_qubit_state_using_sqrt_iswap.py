from typing import List, TYPE_CHECKING
import numpy as np
from cirq import ops, qis, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def prepare_two_qubit_state_using_sqrt_iswap(q0: 'cirq.Qid', q1: 'cirq.Qid', state: 'cirq.STATE_VECTOR_LIKE', *, use_sqrt_iswap_inv: bool=True) -> List['cirq.Operation']:
    """Prepares the given 2q state from |00> using at-most 1 √iSWAP gate + single qubit rotations.

    Entangled states are prepared using exactly 1 √iSWAP gate while product states are prepared
    using only single qubit rotations (0 √iSWAP gates)

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.
        use_sqrt_iswap_inv: If True, uses `cirq.SQRT_ISWAP_INV` instead of `cirq.SQRT_ISWAP`.

    Returns:
        List of operations (at-most 1 √iSWAP + single qubit rotations) preparing `state` from |00>.
    """
    state_vector = qis.to_valid_state_vector(state, num_qubits=2)
    state_vector = state_vector / np.linalg.norm(state_vector)
    u, s, vh = np.linalg.svd(state_vector.reshape(2, 2))
    if np.isclose(s[0], 1):
        return _1q_matrices_to_ops(u, vh.T, q0, q1, True)
    alpha = np.arccos(np.sqrt(np.clip(1 - s[0] * 2 * s[1], 0, 1)))
    sqrt_iswap_gate = ops.SQRT_ISWAP_INV if use_sqrt_iswap_inv else ops.SQRT_ISWAP
    op_list = [ops.ry(2 * alpha).on(q0), sqrt_iswap_gate.on(q0, q1)]
    intermediate_state = circuits.Circuit(op_list).final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64)
    u_iSWAP, _, vh_iSWAP = np.linalg.svd(intermediate_state.reshape(2, 2))
    return op_list + _1q_matrices_to_ops(np.dot(u, np.linalg.inv(u_iSWAP)), np.dot(vh.T, np.linalg.inv(vh_iSWAP.T)), q0, q1)