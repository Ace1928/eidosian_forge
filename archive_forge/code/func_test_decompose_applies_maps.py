import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_decompose_applies_maps():
    a, b, c = cirq.LineQubit.range(3)
    exp = sympy.Symbol('exp')
    theta = sympy.Symbol('theta')
    circuit = cirq.FrozenCircuit(cirq.X(a) ** theta, cirq.Y(b), cirq.H(c), cirq.CX(a, b) ** exp, cirq.measure(a, b, c, key='m'))
    op = cirq.CircuitOperation(circuit=circuit, qubit_map={c: b, b: c}, measurement_key_map={'m': 'p'}, param_resolver={exp: theta, theta: exp})
    expected_circuit = cirq.Circuit(cirq.X(a) ** exp, cirq.Y(c), cirq.H(b), cirq.CX(a, c) ** theta, cirq.measure(a, c, b, key='p'))
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit