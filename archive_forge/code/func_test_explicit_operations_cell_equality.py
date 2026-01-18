import pytest
import cirq
from cirq.interop.quirk.cells.cell import Cell, ExplicitOperationsCell
def test_explicit_operations_cell_equality():
    a = cirq.LineQubit(0)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(ExplicitOperationsCell([], []), ExplicitOperationsCell([]))
    eq.add_equality_group(ExplicitOperationsCell([cirq.X(a)], []))
    eq.add_equality_group(ExplicitOperationsCell([], [cirq.Y(a)]))