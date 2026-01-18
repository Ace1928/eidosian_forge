import cirq
def test_empty_moment():
    circuit = cirq.Circuit([])
    circuit = cirq.expand_composite(circuit)
    assert_equal_mod_empty(cirq.Circuit([]), circuit)