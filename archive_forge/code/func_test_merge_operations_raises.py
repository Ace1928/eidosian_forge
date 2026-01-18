from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_raises():
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.CZ(*q[:2]), cirq.X(q[0]))
    with pytest.raises(ValueError, match='must act on a subset of qubits'):
        cirq.merge_operations(c, lambda *_: cirq.X(q[2]))