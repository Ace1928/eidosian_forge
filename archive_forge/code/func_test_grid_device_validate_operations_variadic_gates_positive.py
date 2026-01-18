from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
@pytest.mark.parametrize('gate_func', [lambda _: cirq.measure, lambda num_qubits: cirq.WaitGate(cirq.Duration(nanos=1), num_qubits=num_qubits)])
def test_grid_device_validate_operations_variadic_gates_positive(gate_func):
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    for q in device_info.grid_qubits:
        device.validate_operation(gate_func(1)(q))
    for i in range(GRID_HEIGHT):
        device.validate_operation(gate_func(2)(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * i + 1]))
    for i in range(GRID_HEIGHT - 1):
        device.validate_operation(gate_func(2)(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * (i + 1)]))
        device.validate_operation(gate_func(2)(device_info.grid_qubits[2 * i + 1], device_info.grid_qubits[2 * (i + 1) + 1]))
    for i in range(GRID_HEIGHT - 2):
        device.validate_operation(gate_func(3)(device_info.grid_qubits[2 * i], device_info.grid_qubits[2 * (i + 1)], device_info.grid_qubits[2 * (i + 2)]))
        device.validate_operation(gate_func(3)(device_info.grid_qubits[2 * i + 1], device_info.grid_qubits[2 * (i + 1) + 1], device_info.grid_qubits[2 * (i + 2) + 1]))
    device.validate_operation(gate_func(len(device_info.grid_qubits))(*device_info.grid_qubits))