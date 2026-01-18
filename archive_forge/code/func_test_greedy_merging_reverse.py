import pytest
import cirq
def test_greedy_merging_reverse():
    """Same as the above test, except that the aligning is done in reverse."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    input_circuit = cirq.Circuit(cirq.Moment([cirq.SWAP(q1, q2), cirq.SWAP(q3, q4)]), cirq.Moment([cirq.X(q4)]), cirq.Moment([cirq.SWAP(q3, q4)]), cirq.Moment([cirq.X(q1)]))
    expected = cirq.Circuit(cirq.Moment([cirq.SWAP(q1, q2), cirq.SWAP(q3, q4)]), cirq.Moment([cirq.X(q1), cirq.X(q4)]), cirq.Moment([cirq.SWAP(q3, q4)]))
    cirq.testing.assert_same_circuits(cirq.stratified_circuit(input_circuit, categories=[cirq.X]), expected)