import pytest
import cirq
from cirq.interop.quirk.cells.cell import Cell, ExplicitOperationsCell
def test_explicit_operations_cell():
    a, b = cirq.LineQubit.range(2)
    v = ExplicitOperationsCell([cirq.X(a)], [cirq.S(a)])
    assert v.operations() == (cirq.X(a),)
    assert v.basis_change() == (cirq.S(a),)
    assert v.controlled_by(b) == ExplicitOperationsCell([cirq.X(a).controlled_by(b)], [cirq.S(a)])