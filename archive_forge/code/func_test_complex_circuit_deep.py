import pytest
import cirq
def test_complex_circuit_deep():
    q = cirq.LineQubit.range(5)
    c_nested = cirq.FrozenCircuit(cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.ISWAP(q[1], q[2]).with_tags('ignore'), cirq.Z(q[4])), cirq.Moment(cirq.Z(q[1]), cirq.ISWAP(q[3], q[4])), cirq.Moment(cirq.ISWAP(q[0], q[1]), cirq.X(q[3])), cirq.Moment(cirq.X.on_each(q[0])))
    c_nested_stratified = cirq.FrozenCircuit(cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.ISWAP(q[1], q[2]).with_tags('ignore')), cirq.Moment(cirq.Z.on_each(q[1], q[4])), cirq.Moment(cirq.ISWAP(*q[:2]), cirq.ISWAP(*q[3:])), cirq.Moment(cirq.X.on_each(q[0], q[3])))
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(5).with_tags('ignore'), c_nested, cirq.CircuitOperation(c_nested).repeat(6).with_tags('preserve_tag'), c_nested)
    c_expected = cirq.Circuit(c_nested_stratified, cirq.CircuitOperation(c_nested).repeat(5).with_tags('ignore'), c_nested_stratified, cirq.CircuitOperation(c_nested_stratified).repeat(6).with_tags('preserve_tag'), c_nested_stratified)
    context = cirq.TransformerContext(tags_to_ignore=['ignore'], deep=True)
    c_stratified = cirq.stratified_circuit(c_orig, context=context, categories=[cirq.X, cirq.Z])
    cirq.testing.assert_same_circuits(c_stratified, c_expected)