from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_moments_drop_empty_moments_deep():
    op = cirq.X(cirq.NamedQubit('q'))
    c_nested = cirq.FrozenCircuit(cirq.Moment(op), cirq.Moment(), cirq.Moment(op))
    circuit_op = cirq.CircuitOperation(c_nested).repeat(2)
    circuit_op_dropped = cirq.CircuitOperation(cirq.FrozenCircuit([op, op])).repeat(2)
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(6).with_tags('ignore'), c_nested, cirq.CircuitOperation(cirq.FrozenCircuit(circuit_op, circuit_op.with_tags('ignore'), circuit_op)).repeat(5).with_tags('preserve_tag'))
    c_expected = cirq.Circuit([op, op], cirq.CircuitOperation(c_nested).repeat(6).with_tags('ignore'), [op, op], cirq.CircuitOperation(cirq.FrozenCircuit(circuit_op_dropped, circuit_op.with_tags('ignore'), circuit_op_dropped)).repeat(5).with_tags('preserve_tag'))
    c_mapped = cirq.map_moments(c_orig, lambda m, i: [] if len(m) == 0 else [m], deep=True, tags_to_ignore=('ignore',))
    cirq.testing.assert_same_circuits(c_mapped, c_expected)