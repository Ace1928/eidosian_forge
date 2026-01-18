import pytest
import cirq
def test_no_categories_earliest_insert():
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    input_circuit = cirq.Circuit(cirq.Moment([cirq.ISWAP(q2, q3)]), cirq.Moment([cirq.X(q1), cirq.ISWAP(q4, q5)]), cirq.Moment([cirq.ISWAP(q1, q2), cirq.X(q4)]))
    cirq.testing.assert_same_circuits(cirq.Circuit(input_circuit.all_operations()), cirq.stratified_circuit(input_circuit))