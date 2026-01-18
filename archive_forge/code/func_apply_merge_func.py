from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def apply_merge_func(op1: ops.Operation, op2: ops.Operation) -> Optional[ops.Operation]:
    if not all((tags_to_ignore_set.isdisjoint(op.tags) for op in [op1, op2])):
        return None
    new_op = merge_func(op1, op2)
    qubit_set = frozenset(op1.qubits + op2.qubits)
    if new_op is not None and (not qubit_set.issuperset(new_op.qubits)):
        raise ValueError(f'Merged operation {new_op} must act on a subset of qubits of original operations {op1} and {op2}')
    return new_op