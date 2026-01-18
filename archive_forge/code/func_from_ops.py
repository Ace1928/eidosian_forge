from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
@staticmethod
def from_ops(*operations: 'cirq.OP_TREE', can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool]=_disjoint_qubits) -> 'CircuitDag':
    dag = CircuitDag(can_reorder=can_reorder)
    for op in ops.flatten_op_tree(operations):
        dag.append(cast(ops.Operation, op))
    return dag