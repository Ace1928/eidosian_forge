import cirq
import cirq.contrib.acquaintance as cca
def test_circular_shift_gate_wire_symbols():
    qubits = [cirq.NamedQubit(q) for q in 'xyz']
    circuit = cirq.Circuit(cca.CircularShiftGate(3, 2)(*qubits))
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = '\nx: ───╲0╱───\n      │\ny: ───╲1╱───\n      │\nz: ───╱2╲───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram
    actual_text_diagram = circuit.to_text_diagram(use_unicode_characters=False)
    expected_text_diagram = '\nx: ---\\0/---\n      |\ny: ---\\1/---\n      |\nz: ---/2\\---\n    '.strip()
    assert actual_text_diagram.strip() == expected_text_diagram