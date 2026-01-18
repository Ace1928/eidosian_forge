import cirq
def test_multiple_composite_default():
    q0, q1 = cirq.LineQubit.range(2)
    cnot = cirq.CNOT(q0, q1)
    circuit = cirq.Circuit()
    circuit.append([cnot, cnot])
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit()
    decomp = [cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5]
    expected.append([decomp, decomp])
    assert_equal_mod_empty(expected, circuit)