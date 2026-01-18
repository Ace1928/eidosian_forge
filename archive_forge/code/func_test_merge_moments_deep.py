from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_moments_deep():
    q = cirq.LineQubit.range(3)
    c_z_moments = cirq.Circuit([cirq.Z.on_each(q[0], q[1]), cirq.Z.on_each(q[1], q[2]), cirq.Z.on_each(q[1], q[0])], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    merged_z_moment = cirq.Moment(cirq.Z.on_each(*q[1:]))
    c_nested_circuit = cirq.FrozenCircuit(c_z_moments, cirq.CCX(*q), c_z_moments)
    c_merged_circuit = cirq.FrozenCircuit(merged_z_moment, cirq.CCX(*q), merged_z_moment)
    c_orig = cirq.Circuit(cirq.CircuitOperation(c_nested_circuit).repeat(5).with_tags('ignore'), c_nested_circuit, cirq.CircuitOperation(c_nested_circuit).repeat(6).with_tags('preserve_tag'), c_nested_circuit, cirq.CircuitOperation(c_nested_circuit).repeat(7))
    c_expected = cirq.Circuit(cirq.CircuitOperation(c_nested_circuit).repeat(5).with_tags('ignore'), c_merged_circuit, cirq.CircuitOperation(c_merged_circuit).repeat(6).with_tags('preserve_tag'), c_merged_circuit, cirq.CircuitOperation(c_merged_circuit).repeat(7))
    cirq.testing.assert_same_circuits(cirq.merge_moments(c_orig, _merge_z_moments_func, tags_to_ignore=('ignore',), deep=True), c_expected)