import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_interchangeable_qubit_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.CZ(a, b), cirq.CZ(b, a))
    eq.add_equality_group(cirq.CZ(a, c))
    eq.add_equality_group(cirq.CNOT(a, b))
    eq.add_equality_group(cirq.CNOT(b, a))
    eq.add_equality_group(cirq.CNOT(a, c))