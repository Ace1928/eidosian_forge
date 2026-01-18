from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def map_operations_and_unroll(circuit: CIRCUIT_TYPE, map_func: Callable[[ops.Operation, int], ops.OP_TREE], *, deep: bool=False, raise_if_add_qubits=True, tags_to_ignore: Sequence[Hashable]=()) -> CIRCUIT_TYPE:
    """Applies local transformations via `cirq.map_operations` & unrolls intermediate circuit ops.

    See `cirq.map_operations` and `cirq.unroll_circuit_op` for more details.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if
            `map_func(op, idx)` adds operations on qubits outside `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.

    Returns:
        Copy of input circuit with mapped operations, unrolled in a moment preserving way.
    """
    return _map_operations_impl(circuit, map_func, deep=deep, raise_if_add_qubits=raise_if_add_qubits, tags_to_ignore=tags_to_ignore, wrap_in_circuit_op=False)