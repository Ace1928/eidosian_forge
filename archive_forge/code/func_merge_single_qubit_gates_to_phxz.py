from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers import transformer_api, transformer_primitives, merge_k_qubit_gates
@transformer_api.transformer
def merge_single_qubit_gates_to_phxz(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, atol: float=1e-08) -> 'cirq.Circuit':
    """Replaces runs of single qubit rotations with a single optional `cirq.PhasedXZGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedXZ.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """

    def rewriter(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        u = protocols.unitary(op)
        if protocols.num_qubits(op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol)
        return gate(op.qubits[0]) if gate else []
    return merge_k_qubit_gates.merge_k_qubit_unitaries(circuit, k=1, context=context, rewriter=rewriter)