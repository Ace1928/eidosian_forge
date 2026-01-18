import cirq
def test_align_right_deep():
    q1, q2 = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Y(q1), cirq.X(q2).with_tags('nocompile')]), cirq.Moment([cirq.X(q2)]), cirq.Moment([cirq.Y(q1)]), cirq.measure(q1, key='a'), cirq.Z(q2).with_classical_controls('a'))
    c_nested_aligned = cirq.FrozenCircuit(cirq.Moment([cirq.X(q1), cirq.X(q2).with_tags('nocompile')]), [cirq.Y(q1), cirq.Y(q1)], cirq.Moment(cirq.measure(q1, key='a'), cirq.X(q2)), cirq.Z(q2).with_classical_controls('a'))
    c_orig = cirq.Circuit(c_nested, cirq.CircuitOperation(c_nested).repeat(6).with_tags('nocompile'), c_nested, cirq.CircuitOperation(c_nested).repeat(5).with_tags('preserve_tag'))
    c_expected = cirq.Circuit(c_nested_aligned, cirq.CircuitOperation(c_nested).repeat(6).with_tags('nocompile'), cirq.Moment(), c_nested_aligned, cirq.CircuitOperation(c_nested_aligned).repeat(5).with_tags('preserve_tag'))
    context = cirq.TransformerContext(tags_to_ignore=['nocompile'], deep=True)
    cirq.testing.assert_same_circuits(cirq.align_right(c_orig, context=context), c_expected)