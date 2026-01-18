from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_can_drop_operations():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(q[0]), cirq.Y(q[1]), cirq.X(q[1]), cirq.Y(q[0]))
    c_mapped = cirq.map_operations(c, lambda op, _: op if op.gate == cirq.X else [])
    c_expected = cirq.Circuit(cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.X(q[1])))
    cirq.testing.assert_same_circuits(c_mapped, c_expected)