from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers import transformer_api, transformer_primitives, merge_k_qubit_gates
@transformer_api.transformer
def merge_single_qubit_moments_to_phxz(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, atol: float=1e-08) -> 'cirq.Circuit':
    """Merges adjacent moments with only 1-qubit rotations to a single moment with PhasedXZ gates.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """
    tags_to_ignore = set(context.tags_to_ignore) if context else set()

    def can_merge_moment(m: 'cirq.Moment'):
        return all((protocols.num_qubits(op) == 1 and protocols.has_unitary(op) and tags_to_ignore.isdisjoint(op.tags) for op in m))

    def merge_func(m1: 'cirq.Moment', m2: 'cirq.Moment') -> Optional['cirq.Moment']:
        if not (can_merge_moment(m1) and can_merge_moment(m2)):
            return None
        ret_ops = []
        for q in m1.qubits | m2.qubits:
            op1, op2 = (m1.operation_at(q), m2.operation_at(q))
            if op1 and op2:
                mat = protocols.unitary(op2) @ protocols.unitary(op1)
                gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(mat, atol)
                if gate:
                    ret_ops.append(gate(q))
            else:
                op = op1 or op2
                assert op is not None
                if isinstance(op.gate, ops.PhasedXZGate):
                    ret_ops.append(op)
                else:
                    gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(protocols.unitary(op), atol)
                    if gate:
                        ret_ops.append(gate(q))
        return circuits.Moment(ret_ops)
    return transformer_primitives.merge_moments(circuit, merge_func, deep=context.deep if context else False, tags_to_ignore=tuple(tags_to_ignore)).unfreeze(copy=False)