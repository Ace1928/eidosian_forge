import pytest
import cirq
def test_heterogeneous_circuit():
    """Tests that a circuit that is very heterogeneous is correctly optimized"""
    q1, q2, q3, q4, q5, q6 = cirq.LineQubit.range(6)
    input_circuit = cirq.Circuit(cirq.Moment([cirq.X(q1), cirq.X(q2), cirq.ISWAP(q3, q4), cirq.ISWAP(q5, q6)]), cirq.Moment([cirq.ISWAP(q1, q2), cirq.ISWAP(q3, q4), cirq.X(q5), cirq.X(q6)]), cirq.Moment([cirq.X(q1), cirq.Z(q2), cirq.X(q3), cirq.Z(q4), cirq.X(q5), cirq.Z(q6)]))
    expected = cirq.Circuit(cirq.Moment([cirq.ISWAP(q3, q4), cirq.ISWAP(q5, q6)]), cirq.Moment([cirq.X(q1), cirq.X(q2), cirq.X(q5), cirq.X(q6)]), cirq.Moment([cirq.ISWAP(q1, q2), cirq.ISWAP(q3, q4)]), cirq.Moment([cirq.Z(q2), cirq.Z(q4), cirq.Z(q6)]), cirq.Moment([cirq.X(q1), cirq.X(q3), cirq.X(q5)]))
    cirq.testing.assert_same_circuits(cirq.stratified_circuit(input_circuit, categories=[cirq.X, cirq.Z]), expected)