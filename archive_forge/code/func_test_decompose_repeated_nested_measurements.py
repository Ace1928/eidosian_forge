import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_decompose_repeated_nested_measurements():
    a = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='A'))).with_measurement_key_mapping({'A': 'B'}).repeat(2, ['zero', 'one'])
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='P'), op1)).with_measurement_key_mapping({'B': 'C', 'P': 'Q'}).repeat(2, ['zero', 'one'])
    op3 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='X'), op2)).with_measurement_key_mapping({'C': 'D', 'X': 'Y'}).repeat(2, ['zero', 'one'])
    expected_measurement_keys_in_order = ['zero:Y', 'zero:zero:Q', 'zero:zero:zero:D', 'zero:zero:one:D', 'zero:one:Q', 'zero:one:zero:D', 'zero:one:one:D', 'one:Y', 'one:zero:Q', 'one:zero:zero:D', 'one:zero:one:D', 'one:one:Q', 'one:one:zero:D', 'one:one:one:D']
    assert cirq.measurement_key_names(op3) == set(expected_measurement_keys_in_order)
    expected_circuit = cirq.Circuit()
    for key in expected_measurement_keys_in_order:
        expected_circuit.append(cirq.measure(a, key=cirq.MeasurementKey.parse_serialized(key)))
    assert cirq.Circuit(cirq.decompose(op3)) == expected_circuit
    assert cirq.measurement_key_names(expected_circuit) == set(expected_measurement_keys_in_order)
    assert op3.mapped_circuit(deep=True) == expected_circuit