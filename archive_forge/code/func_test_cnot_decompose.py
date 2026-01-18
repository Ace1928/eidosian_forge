import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_cnot_decompose():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.decompose_once(cirq.CNOT(a, b) ** sympy.Symbol('x')) is not None