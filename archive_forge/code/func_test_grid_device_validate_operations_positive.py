from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_grid_device_validate_operations_positive():
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    variadic_gates = [cirq.measure, cirq.WaitGate(cirq.Duration(nanos=1), num_qubits=2)]
    for q in device_info.grid_qubits:
        device.validate_operation(cirq.X(q))
        device.validate_operation(cirq.measure(q))
    for i in range(GRID_HEIGHT):
        device.validate_operation(cirq.CZ(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * i + 1]))
        for gate in variadic_gates:
            device.validate_operation(gate(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * i + 1]))