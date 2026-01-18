from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def merge_operations(circuit: CIRCUIT_TYPE, merge_func: Callable[[ops.Operation, ops.Operation], Optional[ops.Operation]], *, tags_to_ignore: Sequence[Hashable]=(), deep: bool=False) -> CIRCUIT_TYPE:
    """Merges operations in a circuit by calling `merge_func` iteratively on operations.

    Two operations op1 and op2 are merge-able if
        - There is no other operations between op1 and op2 in the circuit
        - is_subset(op1.qubits, op2.qubits) or is_subset(op2.qubits, op1.qubits)

    The `merge_func` is a callable which, given two merge-able operations
    op1 and op2, decides whether they should be merged into a single operation
    or not. If not, it should return None, else it should return the single merged
    operations `op`.

    The method iterates on the input circuit moment-by-moment from left to right and attempts
    to repeatedly merge each operation in the latest moment with all the corresponding merge-able
    operations to its left.

    If op1 and op2 are merged, both op1 and op2 are deleted from the circuit and
    the resulting `merged_op` is inserted at the index corresponding to the larger
    of op1/op2. If both op1 and op2 act on the same number of qubits, `merged_op` is
    inserted in the smaller moment index to minimize circuit depth.

    The number of calls to `merge_func` is O(N), where N = Total no. of operations, because:
        - Every time the `merge_func` returns a new operation, the number of operations in the
            circuit reduce by 1 and hence this can happen at most O(N) times
        - Every time the `merge_func` returns None, the current operation is inserted into the
            frontier and we go on to process the next operation, which can also happen at-most
            O(N) times.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        merge_func: Callable to determine whether two merge-able operations in the circuit should
            be merged. If the operations can be merged, the callable should return the merged
            operation, else None.
        tags_to_ignore: Sequence of tags which should be ignored while applying `merge_func` on
            tagged operations -- i.e. `merge_func(op1, op2)` will be called only if both `op1` and
            `op2` satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.


    Returns:
        Copy of input circuit with merged operations.

    Raises:
        ValueError if the merged operation acts on new qubits outside the set of qubits
            corresponding to the original operations to be merged.
    """
    _circuit_op_tag = '_internal_tag_to_mark_circuit_ops_in_circuit'
    tags_to_ignore_set = set(tags_to_ignore) | {_circuit_op_tag}

    def apply_merge_func(op1: ops.Operation, op2: ops.Operation) -> Optional[ops.Operation]:
        if not all((tags_to_ignore_set.isdisjoint(op.tags) for op in [op1, op2])):
            return None
        new_op = merge_func(op1, op2)
        qubit_set = frozenset(op1.qubits + op2.qubits)
        if new_op is not None and (not qubit_set.issuperset(new_op.qubits)):
            raise ValueError(f'Merged operation {new_op} must act on a subset of qubits of original operations {op1} and {op2}')
        return new_op
    merged_circuit = _MergedCircuit()
    for moment_idx, current_moment in enumerate(cast(List['cirq.Moment'], circuit)):
        merged_circuit.append_empty_moment()
        for op in sorted(current_moment.operations, key=lambda op: op.qubits):
            if deep and isinstance(op.untagged, circuits.CircuitOperation) and tags_to_ignore_set.isdisjoint(op.tags):
                op_untagged = op.untagged
                merged_circuit.add_op_to_moment(moment_idx, op_untagged.replace(circuit=merge_operations(op_untagged.circuit, merge_func, tags_to_ignore=tags_to_ignore, deep=True)).with_tags(*op.tags, _circuit_op_tag))
                continue
            op_qs = set(op.qubits)
            left_idx, left_ops = merged_circuit.get_mergeable_ops(op, op_qs)
            if len(left_ops) == 1 and op_qs.issubset(left_ops[0].qubits):
                new_op = apply_merge_func(left_ops[0], op)
                if new_op is not None:
                    merged_circuit.remove_op_from_moment(left_idx, left_ops[0])
                    merged_circuit.add_op_to_moment(left_idx, new_op)
                else:
                    merged_circuit.add_op_to_moment(moment_idx, op)
                continue
            while left_ops and op_qs:
                for left_op in left_ops:
                    is_merged = False
                    if op_qs.issuperset(left_op.qubits):
                        new_op = apply_merge_func(left_op, op)
                        if new_op is not None:
                            merged_circuit.remove_op_from_moment(left_idx, left_op)
                            op, is_merged = (new_op, True)
                    if not is_merged:
                        op_qs -= frozenset(left_op.qubits)
                left_idx, left_ops = merged_circuit.get_mergeable_ops(op, op_qs)
            merged_circuit.add_op_to_moment(moment_idx, op)
    ret_circuit = merged_circuit.get_cirq_circuit()
    if deep:
        ret_circuit = map_operations(ret_circuit, lambda o, _: o.untagged.with_tags(*set(o.tags) - {_circuit_op_tag}), deep=True)
    return _to_target_circuit_type(ret_circuit, circuit)