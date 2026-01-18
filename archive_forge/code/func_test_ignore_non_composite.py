import cirq
def test_ignore_non_composite():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.Y(q1), cirq.CZ(q0, q1), cirq.Z(q0)])
    expected = circuit.copy()
    circuit = cirq.expand_composite(circuit)
    assert_equal_mod_empty(expected, circuit)