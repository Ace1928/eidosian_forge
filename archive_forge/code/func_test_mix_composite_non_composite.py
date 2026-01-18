import cirq
def test_mix_composite_non_composite():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.X(q1))
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(cirq.X(q0), cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5, cirq.X(q1), strategy=cirq.InsertStrategy.NEW)
    assert_equal_mod_empty(expected, circuit)