import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_invalid_measurement_keys():
    a = cirq.LineQubit(0)
    circuit = cirq.FrozenCircuit(cirq.measure(a, key='m'))
    c_op = cirq.CircuitOperation(circuit)
    with pytest.raises(ValueError, match='Mapping to invalid key: m:a'):
        _ = c_op.with_measurement_key_mapping({'m': 'm:a'})
    with pytest.raises(ValueError, match='Mapping to invalid key: m:a'):
        _ = cirq.CircuitOperation(cirq.FrozenCircuit(c_op), measurement_key_map={'m': 'm:a'})
    with pytest.raises(ValueError, match='Invalid key name: m:a'):
        _ = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='m:a')))
    _ = cirq.CircuitOperation(circuit, measurement_key_map={'m:a': 'ma'})