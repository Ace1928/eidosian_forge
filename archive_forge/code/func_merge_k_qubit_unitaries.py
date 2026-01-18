from typing import cast, Optional, Callable, TYPE_CHECKING
from cirq import ops, protocols, circuits
from cirq.transformers import transformer_api, transformer_primitives
@transformer_api.transformer
def merge_k_qubit_unitaries(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, k: int=0, rewriter: Optional[Callable[['cirq.CircuitOperation'], 'cirq.OP_TREE']]=None) -> 'cirq.Circuit':
    """Merges connected components of unitary operations, acting on <= k qubits.

    Uses rewriter to convert a connected component of unitary operations acting on <= k-qubits
    into a more desirable form. If not specified, connected components are replaced by a single
    `cirq.MatrixGate` containing unitary matrix of the merged component.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        k: Connected components of unitary operations acting on <= k qubits are merged.
        rewriter: Callable type that takes a `cirq.CircuitOperation`, encapsulating a connected
            component of unitary operations acting on <= k qubits, and produces a `cirq.OP_TREE`.
            Specifies how to merge the connected component into a more desirable form.

    Returns:
        Copy of the transformed input circuit.

    Raises:
        ValueError: If k <= 0
    """
    if k <= 0:
        raise ValueError(f'k should be greater than or equal to 1. Found {k}.')
    merged_circuit_op_tag = '_merged_k_qubit_unitaries_component'
    circuit = transformer_primitives.merge_k_qubit_unitaries_to_circuit_op(circuit, k=k, tags_to_ignore=context.tags_to_ignore if context else (), merged_circuit_op_tag=merged_circuit_op_tag, deep=context.deep if context else False)
    return _rewrite_merged_k_qubit_unitaries(circuit, context=context, k=k, rewriter=rewriter, merged_circuit_op_tag=merged_circuit_op_tag)