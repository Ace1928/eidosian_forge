from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def map_operations(circuit: CIRCUIT_TYPE, map_func: Callable[[ops.Operation, int], ops.OP_TREE], *, deep: bool=False, raise_if_add_qubits=True, tags_to_ignore: Sequence[Hashable]=()) -> CIRCUIT_TYPE:
    """Applies local transformations, by calling `map_func(op, moment_index)` for each operation.

    By default, the function assumes `issubset(qubit_set(map_func(op, moment_index)), op.qubits)` is
    True.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE. If the
            resulting optree spans more than 1 moment, it's inserted in-place in the same moment as
            `cirq.CircuitOperation(cirq.FrozenCircuit(op_tree)).with_tags(MAPPED_CIRCUIT_OP_TAG)`
            to preserve moment structure. Utility methods like `cirq.unroll_circuit_op` can
            subsequently be used to unroll the mapped circuit operation.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if
            `map_func(op, idx)` adds operations on qubits outside of `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.

    Raises:
          ValueError if `issubset(qubit_set(map_func(op, idx)), op.qubits) is False` and
            `raise_if_add_qubits is True`.

    Returns:
        Copy of input circuit with mapped operations (wrapped in a tagged CircuitOperation).
    """
    return _map_operations_impl(circuit, map_func, deep=deep, raise_if_add_qubits=raise_if_add_qubits, tags_to_ignore=tags_to_ignore, wrap_in_circuit_op=True)