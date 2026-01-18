import cirq
def test_align_basic_no_context():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    c = cirq.Circuit([cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Y(q1), cirq.X(q2)]), cirq.Moment([cirq.X(q1)])])
    cirq.testing.assert_same_circuits(cirq.align_left(c), cirq.Circuit(cirq.Moment([cirq.X(q1), cirq.X(q2)]), cirq.Moment([cirq.Y(q1)]), cirq.Moment([cirq.X(q1)])))
    cirq.testing.assert_same_circuits(cirq.align_right(c), cirq.Circuit(cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.Y(q1)]), cirq.Moment([cirq.X(q1), cirq.X(q2)])))