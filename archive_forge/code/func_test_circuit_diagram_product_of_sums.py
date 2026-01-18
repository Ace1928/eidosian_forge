from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_circuit_diagram_product_of_sums():
    qubits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledGate(MultiH(2))(*qubits))
    cirq.testing.assert_has_diagram(c, '\n0: ───@─────────\n      │\n1: ───H(q(1))───\n      │\n2: ───H(q(2))───\n')
    qubits = cirq.LineQid.for_qid_shape((3, 3, 3, 2))
    c = cirq.Circuit(MultiH(1)(*qubits[3:]).controlled_by(*qubits[:3], control_values=[1, (0, 1), (2, 0)]))
    cirq.testing.assert_has_diagram(c, '\n0 (d=3): ───@───────────────\n            │\n1 (d=3): ───(0,1)───────────\n            │\n2 (d=3): ───(0,2)───────────\n            │\n3 (d=2): ───H(q(3) (d=2))───\n')