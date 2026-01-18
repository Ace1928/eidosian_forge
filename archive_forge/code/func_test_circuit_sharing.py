import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_circuit_sharing():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(cirq.X(a), cirq.Y(b), cirq.H(c), cirq.CX(a, b) ** sympy.Symbol('exp'), cirq.measure(a, b, c, key='m'))
    op1 = cirq.CircuitOperation(circuit)
    op2 = cirq.CircuitOperation(op1.circuit)
    op3 = circuit.to_op()
    assert op1.circuit is circuit
    assert op2.circuit is circuit
    assert op3.circuit is circuit
    assert hash(op1) == hash(op2)
    assert hash(op1) == hash(op3)