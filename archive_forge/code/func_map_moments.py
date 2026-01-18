from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def map_moments(circuit: CIRCUIT_TYPE, map_func: Callable[[circuits.Moment, int], Union[circuits.Moment, Sequence[circuits.Moment]]], *, tags_to_ignore: Sequence[Hashable]=(), deep: bool=False) -> CIRCUIT_TYPE:
    """Applies local transformation on moments, by calling `map_func(moment)` for each moment.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Moment, moment_index) to a sequence of moments.
        tags_to_ignore: Tagged circuit operations marked with any of `tags_to_ignore` will be
            ignored when recursively applying the transformer primitive to sub-circuits, given
            deep=True.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.

    Returns:
        Copy of input circuit with mapped moments.
    """
    mutable_circuit = circuit.unfreeze(copy=False)
    if deep:
        batch_replace = []
        for i, op in circuit.findall_operations(lambda o: isinstance(o.untagged, circuits.CircuitOperation)):
            if set(op.tags).intersection(tags_to_ignore):
                continue
            op_untagged = cast(circuits.CircuitOperation, op.untagged)
            mapped_op = op_untagged.replace(circuit=map_moments(op_untagged.circuit, map_func, tags_to_ignore=tags_to_ignore, deep=deep)).with_tags(*op.tags)
            batch_replace.append((i, op, mapped_op))
        mutable_circuit = circuit.unfreeze(copy=True)
        mutable_circuit.batch_replace(batch_replace)
    return _create_target_circuit_type((map_func(mutable_circuit[i], i) for i in range(len(mutable_circuit))), circuit)