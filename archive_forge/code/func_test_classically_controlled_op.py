import cirq
def test_classically_controlled_op():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key='m'), cirq.X(q1).with_classical_controls('m'))
    assert_optimizes(before=circuit, after=circuit)