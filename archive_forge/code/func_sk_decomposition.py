import math
import warnings
from functools import lru_cache
from scipy.spatial import KDTree
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
def sk_decomposition(op, epsilon, *, max_depth=5, basis_set=('T', 'T*', 'H'), basis_length=10):
    """Approximate an arbitrary single-qubit gate in the Clifford+T basis using the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    This method implements the Solovay-Kitaev decomposition algorithm that approximates any single-qubit
    operation with :math:`\\epsilon > 0` error. The procedure exits when the approximation error
    becomes less than :math:`\\epsilon`, or when ``max_depth`` approximation passes have been made. In the
    latter case, the approximation error could be :math:`\\geq \\epsilon`.

    This algorithm produces a decomposition with :math:`O(\\text{log}^{3.97}(1/\\epsilon))` operations.

    Args:
        op (~pennylane.operation.Operation): A single-qubit gate operation.
        epsilon (float): The maximum permissible error.

    Keyword Args:
        max_depth (int): The maximum number of approximation passes. A smaller :math:`\\epsilon` would generally require
            a greater number of passes. Default is ``5``.
        basis_set (list[str]): Basis set to be used for the decomposition and building an approximate set internally.
            It accepts the following gate terms: ``['X', 'Y', 'Z', 'H', 'T', 'T*', 'S', 'S*']``, where ``*`` refers
            to the gate adjoint. Default value is ``['T', 'T*', 'H']``.
        basis_length (int): Maximum expansion length of Clifford+T sequences in the internally-built approximate set.
            Default is ``10``.

    Returns:
        list[~pennylane.operation.Operation]: A list of gates in the Clifford+T basis set that approximates the given
        operation along with a final global phase operation. The operations are in the circuit-order.

    Raises:
        ValueError: If the given operator acts on more than one wires.

    **Example**

    Suppose one would like to decompose :class:`~.RZ` with rotation angle :math:`\\phi = \\pi/3`:

    .. code-block:: python3

        import numpy as np
        import pennylane as qml

        op  = qml.RZ(np.pi/3, wires=0)

        # Get the gate decomposition in ['T', 'T*', 'H']
        ops = qml.ops.sk_decomposition(op, epsilon=1e-3)

        # Get the approximate matrix from the ops
        matrix_sk = qml.prod(*reversed(ops)).matrix()

    When the function is run for a sufficient ``depth`` with a good enough approximate set,
    the output gate sequence should implement the same operation approximately.

    >>> qml.math.allclose(op.matrix(), matrix_sk, atol=1e-3)
    True

    """
    if len(op.wires) != 1:
        raise ValueError(f'Operator must be a single qubit operation, got {op} acting on {op.wires} wires.')
    with QueuingManager.stop_recording():
        approx_set_ids, approx_set_mat, approx_set_gph, approx_set_qat = _approximate_set(basis_set, max_length=basis_length)
        kd_tree = KDTree(qml.math.array(approx_set_qat))
        op_matrix = op.matrix()
        interface = qml.math.get_deep_interface(op_matrix)
        gate_mat, gate_gph = _SU2_transform(qml.math.unwrap(op_matrix))
        gate_qat = _quaternion_transform(gate_mat)

        def _solovay_kitaev(umat, n, u_n1_ids, u_n1_mat):
            """Recursive method as given in the Section 3 of arXiv:0505030"""
            if not n:
                seq_node = qml.math.array([_quaternion_transform(umat)])
                _, [index] = kd_tree.query(seq_node, workers=-1)
                return (approx_set_ids[index], approx_set_mat[index])
            v_n, w_n = _group_commutator_decompose(umat @ qml.math.conj(qml.math.transpose(u_n1_mat)))
            c_ids_mats = []
            for c_n in [v_n, w_n]:
                c_n1_ids, c_n1_mat = (None, None)
                for i in range(n):
                    c_n1_ids, c_n1_mat = _solovay_kitaev(c_n, i, c_n1_ids, c_n1_mat)
                c_n1_ids_adj = [qml.adjoint(gate, lazy=False) for gate in reversed(c_n1_ids)]
                c_n1_mat_adj = qml.math.conj(qml.math.transpose(c_n1_mat))
                c_ids_mats.append([c_n1_ids, c_n1_mat, c_n1_ids_adj, c_n1_mat_adj])
            v_n1_ids, v_n1_mat, v_n1_ids_adj, v_n1_mat_adj = c_ids_mats[0]
            w_n1_ids, w_n1_mat, w_n1_ids_adj, w_n1_mat_adj = c_ids_mats[1]
            approx_ids = u_n1_ids + w_n1_ids_adj + v_n1_ids_adj + w_n1_ids + v_n1_ids
            approx_mat = v_n1_mat @ w_n1_mat @ v_n1_mat_adj @ w_n1_mat_adj @ u_n1_mat
            return (approx_ids, approx_mat)
        _, [index] = kd_tree.query(qml.math.array([gate_qat]), workers=-1)
        decomposition, u_prime = (approx_set_ids[index], approx_set_mat[index])
        for depth in range(max_depth):
            if qml.math.norm(gate_mat[0] - u_prime[0]) <= epsilon:
                break
            decomposition, u_prime = _solovay_kitaev(gate_mat, depth + 1, decomposition, u_prime)
        [new_tape], _ = qml.transforms.cancel_inverses(QuantumScript(decomposition or [qml.Identity(0)]))
    [map_tape], _ = qml.map_wires(new_tape, wire_map={0: op.wires[0]}, queue=True)
    phase = approx_set_gph[index] - gate_gph if depth or qml.math.allclose(gate_gph, 0.0) else 0.0
    global_phase = qml.GlobalPhase(qml.math.array(phase, like=interface))
    return map_tape.operations + [global_phase]