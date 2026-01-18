import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_diagram_multiple_ops_single_moment():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'), cirq.X(q0).with_classical_controls('a'), cirq.X(q1).with_classical_controls('b'))
    cirq.testing.assert_has_diagram(circuit, '\n      ┌──┐   ┌──┐\n0: ────M──────X─────\n       ║      ║\n1: ────╫M─────╫X────\n       ║║     ║║\na: ════@╬═════^╬════\n        ║      ║\nb: ═════@══════^════\n      └──┘   └──┘\n', use_unicode_characters=True)