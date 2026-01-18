from typing import Optional, TYPE_CHECKING, Set, List
import pytest
import cirq
from cirq import PointOptimizer, PointOptimizationSummary, Operation
from cirq.testing import EqualsTester
def optimization_at(self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation') -> Optional['cirq.PointOptimizationSummary']:
    new_op = op
    if len(op.qubits) == 1 and isinstance(op, cirq.GateOperation):
        new_op = op.gate(cirq.LineQubit(42))
    return cirq.PointOptimizationSummary(clear_span=1, clear_qubits=op.qubits, new_operations=new_op)