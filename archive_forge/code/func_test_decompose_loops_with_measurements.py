import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_decompose_loops_with_measurements():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b), cirq.measure(a, b, key='m'))
    base_op = cirq.CircuitOperation(circuit)
    op = base_op.with_qubits(b, a).repeat(3)
    expected_circuit = cirq.Circuit(cirq.H(b), cirq.CX(b, a), cirq.measure(b, a, key=cirq.MeasurementKey.parse_serialized('0:m')), cirq.H(b), cirq.CX(b, a), cirq.measure(b, a, key=cirq.MeasurementKey.parse_serialized('1:m')), cirq.H(b), cirq.CX(b, a), cirq.measure(b, a, key=cirq.MeasurementKey.parse_serialized('2:m')))
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit