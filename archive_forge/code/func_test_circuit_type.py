import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_circuit_type():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.X(a), cirq.Y(b), cirq.H(c), cirq.CX(a, b) ** sympy.Symbol('exp'), cirq.measure(a, b, c, key='m'))
    with pytest.raises(TypeError, match='Expected circuit of type FrozenCircuit'):
        _ = cirq.CircuitOperation(circuit)