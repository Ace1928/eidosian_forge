import pytest
import cirq
def test_greedy_merging():
    """Tests a tricky situation where the algorithm of "Merge single-qubit
    gates, greedily align single-qubit then 2-qubit operations" doesn't work.
    Our algorithm succeeds because we also run it in reverse order."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    input_circuit = cirq.Circuit(cirq.Moment([cirq.X(q1)]), cirq.Moment([cirq.SWAP(q1, q2), cirq.SWAP(q3, q4)]), cirq.Moment([cirq.X(q3)]), cirq.Moment([cirq.SWAP(q3, q4)]))
    expected = cirq.Circuit(cirq.Moment([cirq.SWAP(q3, q4)]), cirq.Moment([cirq.X(q1), cirq.X(q3)]), cirq.Moment([cirq.SWAP(q1, q2), cirq.SWAP(q3, q4)]))
    cirq.testing.assert_same_circuits(cirq.stratified_circuit(input_circuit, categories=[cirq.X]), expected)