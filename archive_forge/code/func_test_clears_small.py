import cirq
def test_clears_small():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.Moment(cirq.Z(a) ** 1e-06))
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(circuit, atol=0.001), cirq.Circuit(cirq.Moment()))