import cirq
def test_drop_empty_moments():
    q1, q2 = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(cirq.Moment(), cirq.Moment(), cirq.Moment([cirq.CNOT(q1, q2)]), cirq.Moment())
    c_nested_dropped = cirq.FrozenCircuit(cirq.CNOT(q1, q2))
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(6).with_tags('nocompile'), c_nested, cirq.CircuitOperation(c_nested).repeat(5).with_tags('preserve_tag'), c_nested)
    c_expected = cirq.Circuit(c_nested_dropped, cirq.CircuitOperation(c_nested).repeat(6).with_tags('nocompile'), c_nested_dropped, cirq.CircuitOperation(c_nested_dropped).repeat(5).with_tags('preserve_tag'), c_nested_dropped)
    context = cirq.TransformerContext(tags_to_ignore=('nocompile',), deep=True)
    cirq.testing.assert_same_circuits(cirq.drop_empty_moments(c_orig, context=context), c_expected)