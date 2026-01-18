import cirq
def test_blocked_shift_one():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    before = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.H(q2)]), cirq.Moment([cirq.measure(q1), cirq.Z(q2)]), cirq.Moment([cirq.H(q1), cirq.measure(q2).with_tags(NO_COMPILE_TAG)])])
    after = cirq.Circuit([cirq.Moment([cirq.H(q1), cirq.H(q2)]), cirq.Moment([cirq.measure(q1), cirq.Z(q2)]), cirq.Moment([cirq.H(q1)]), cirq.Moment([cirq.measure(q2).with_tags(NO_COMPILE_TAG)])])
    assert_optimizes(before=before, after=after)
    assert_optimizes(before=before, after=before, with_context=True)