import cirq
def test_decompose_returns_not_flat_op_tree():

    class ExampleGate(cirq.testing.SingleQubitGate):

        def _decompose_(self, qubits):
            q0, = qubits
            yield (cirq.X(q0),)
    q0 = cirq.NamedQubit('q0')
    circuit = cirq.Circuit(ExampleGate()(q0))
    circuit = cirq.expand_composite(circuit)
    expected = cirq.Circuit(cirq.X(q0))
    assert_equal_mod_empty(expected, circuit)