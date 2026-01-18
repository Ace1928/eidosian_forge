import numpy as np
import pytest
import sympy
import cirq
def test_circuit_diagram_info_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.CircuitDiagramInfo(('X',)))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('X', 'Y')), cirq.CircuitDiagramInfo(('X', 'Y'), 1))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 2))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z', 'Z'), 2))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 3))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 3, auto_exponent_parens=False))