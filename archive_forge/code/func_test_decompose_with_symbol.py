import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_decompose_with_symbol():
    q0, = _make_qubits(1)
    ps = cirq.PauliString({q0: cirq.Y})
    op = cirq.PauliStringPhasor(ps, exponent_neg=sympy.Symbol('a'))
    circuit = cirq.Circuit(op)
    circuit = cirq.expand_composite(circuit)
    cirq.testing.assert_has_diagram(circuit, 'q0: ───X^0.5───Z^a───X^-0.5───')
    ps = cirq.PauliString({q0: cirq.Y}, -1)
    op = cirq.PauliStringPhasor(ps, exponent_neg=sympy.Symbol('a'))
    circuit = cirq.Circuit(op)
    circuit = cirq.expand_composite(circuit)
    cirq.testing.assert_has_diagram(circuit, 'q0: ───X^0.5───X───Z^a───X───X^-0.5───')