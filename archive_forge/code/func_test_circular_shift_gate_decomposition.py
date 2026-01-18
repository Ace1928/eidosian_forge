import cirq
import cirq.contrib.acquaintance as cca
def test_circular_shift_gate_decomposition():
    qubits = [cirq.NamedQubit(q) for q in 'abcdef']
    circular_shift = cca.CircularShiftGate(2, 1, cirq.CZ)(*qubits[:2])
    circuit = cirq.expand_composite(cirq.Circuit(circular_shift))
    expected_circuit = cirq.Circuit((cirq.Moment((cirq.CZ(*qubits[:2]),)),))
    assert circuit == expected_circuit
    no_decomp = lambda op: isinstance(op, cirq.GateOperation) and op.gate == cirq.SWAP
    circular_shift = cca.CircularShiftGate(6, 3)(*qubits)
    circuit = cirq.expand_composite(cirq.Circuit(circular_shift), no_decomp=no_decomp)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = '\na: ───────────×───────────\n              │\nb: ───────×───×───×───────\n          │       │\nc: ───×───×───×───×───×───\n      │       │       │\nd: ───×───×───×───×───×───\n          │       │\ne: ───────×───×───×───────\n              │\nf: ───────────×───────────\n    '.strip()
    assert actual_text_diagram == expected_text_diagram
    circular_shift = cca.CircularShiftGate(6, 2)(*qubits)
    circuit = cirq.expand_composite(cirq.Circuit(circular_shift), no_decomp=no_decomp)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = '\na: ───────×───────────────\n          │\nb: ───×───×───×───────────\n      │       │\nc: ───×───×───×───×───────\n          │       │\nd: ───────×───×───×───×───\n              │       │\ne: ───────────×───×───×───\n                  │\nf: ───────────────×───────\n    '.strip()
    assert actual_text_diagram == expected_text_diagram