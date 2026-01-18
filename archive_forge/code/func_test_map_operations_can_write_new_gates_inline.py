from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_can_write_new_gates_inline():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    z = cirq.NamedQubit('z')
    c = cirq.Circuit(cirq.CZ(x, y), cirq.Y(x), cirq.Z(x), cirq.X(y), cirq.CNOT(y, z), cirq.Z(y), cirq.Z(x), cirq.CNOT(y, z), cirq.CNOT(z, y))
    cirq.testing.assert_has_diagram(c, '\nx: ───@───Y───Z───Z───────────\n      │\ny: ───@───X───@───Z───@───X───\n              │       │   │\nz: ───────────X───────X───@───\n')
    expected_diagram = '\nx: ───X───X───X───X───────────\n\ny: ───X───X───X───X───X───X───\n\nz: ───────────X───────X───X───\n'
    cirq.testing.assert_has_diagram(cirq.map_operations(c, lambda op, _: cirq.X.on_each(*op.qubits)), expected_diagram)
    cirq.testing.assert_has_diagram(cirq.map_operations_and_unroll(c, lambda op, _: cirq.X.on_each(*op.qubits)), expected_diagram)