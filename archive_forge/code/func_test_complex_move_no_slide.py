import cirq
def test_complex_move_no_slide():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.H(q2)]), cirq.Moment([cirq.measure(q1), cirq.Z(q2)]), cirq.Moment([cirq.H(q1), cirq.measure(q2).with_tags(NO_COMPILE_TAG)]), cirq.Moment([cirq.H(q3)]), cirq.Moment([cirq.X(q1), cirq.measure(q3)])])
    after = cirq.Circuit([cirq.Moment(cirq.H(q1), cirq.H(q2)), cirq.Moment(cirq.measure(q1), cirq.Z(q2)), cirq.Moment(cirq.H(q1)), cirq.Moment(cirq.H(q3)), cirq.Moment(cirq.X(q1), cirq.measure(q2).with_tags(NO_COMPILE_TAG), cirq.measure(q3))])
    assert_optimizes(before=before, after=after, measure_only_moment=False)
    assert_optimizes(before=before, after=before, measure_only_moment=False, with_context=True)