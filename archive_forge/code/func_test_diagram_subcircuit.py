import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_diagram_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'))))
    cirq.testing.assert_has_diagram(circuit, '\n      [ 0: ───M─────── ]\n      [       ║        ]\n0: ───[ 1: ───╫───X─── ]───\n      [       ║   ║    ]\n      [ a: ═══@═══^═══ ]\n      │\n1: ───#2───────────────────\n', use_unicode_characters=True)