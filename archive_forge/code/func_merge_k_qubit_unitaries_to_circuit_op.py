from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def merge_k_qubit_unitaries_to_circuit_op(circuit: CIRCUIT_TYPE, k: int, *, tags_to_ignore: Sequence[Hashable]=(), merged_circuit_op_tag: Optional[str]=None, deep: bool=False) -> CIRCUIT_TYPE:
    """Merges connected components of operations, acting on <= k qubits, into circuit operations.

    Uses `cirq.merge_operations_to_circuit_op` to identify and merge connected components of
    unitary operations acting on at-most k-qubits. Moment structure is preserved for operations
    that do not participate in merging. For merged operations, the newly created circuit operations
    are constructed by inserting operations using EARLIEST strategy.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        k: Merge-able operations acting on <= k qubits are merged into a connected component.
        tags_to_ignore: Tagged operations marked any of `tags_to_ignore` will not be considered as
            potential candidates for any connected component.
        merged_circuit_op_tag: Tag to be applied on circuit operations wrapping valid connected
            components. A default tag is applied if left None.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.

    Returns:
        Copy of input circuit with valid connected components wrapped in tagged circuit operations.
    """

    def can_merge(ops1: Sequence['cirq.Operation'], ops2: Sequence['cirq.Operation']) -> bool:
        return all((protocols.num_qubits(op) <= k and protocols.has_unitary(op) for op_list in [ops1, ops2] for op in op_list))
    return merge_operations_to_circuit_op(circuit, can_merge, tags_to_ignore=tags_to_ignore, merged_circuit_op_tag=merged_circuit_op_tag or f'Merged {k}q unitary connected component.', deep=deep)