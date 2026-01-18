import numpy as np
import pytest
import sympy
import cirq
def test_ms_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.SWAP(a, b), cirq.X(a), cirq.Y(a), cirq.ms(np.pi).on(a, b))
    cirq.testing.assert_has_diagram(circuit, '\na: ───×───X───Y───MS(π)───\n      │           │\nb: ───×───────────MS(π)───\n')