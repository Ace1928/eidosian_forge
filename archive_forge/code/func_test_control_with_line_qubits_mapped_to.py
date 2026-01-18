import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_control_with_line_qubits_mapped_to():
    a, b = cirq.LineQubit.range(2)
    a2, b2 = cirq.NamedQubit.range(2, prefix='q')
    cell = cirq.interop.quirk.cells.control_cells.ControlCell(a, [cirq.Y(b) ** 0.5])
    mapped_cell = cirq.interop.quirk.cells.control_cells.ControlCell(a2, [cirq.Y(b2) ** 0.5])
    assert cell != mapped_cell
    assert cell.with_line_qubits_mapped_to([a2, b2]) == mapped_cell