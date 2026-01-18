import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_diagram_pauli():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure_single_paulistring(cirq.X(q0), key='a'), cirq.X(q1).with_classical_controls('a'))
    cirq.testing.assert_has_diagram(circuit, '\n0: ───M(X)───────\n      ║\n1: ───╫──────X───\n      ║      ║\na: ═══@══════^═══\n', use_unicode_characters=True)