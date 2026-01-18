import cirq
def test_multi_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q1), cirq.measure(q0, q1, key='m'))
    assert_optimizes(before=circuit, after=circuit)