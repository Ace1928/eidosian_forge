from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_grid_device_from_proto():
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    assert len(device.metadata.qubit_set) == len(device_info.grid_qubits)
    assert device.metadata.qubit_set == frozenset(device_info.grid_qubits)
    assert all((frozenset(pair) in device.metadata.qubit_pairs for pair in device_info.qubit_pairs))
    assert device.metadata.gateset == device_info.expected_gateset
    assert device.metadata.gate_durations == device_info.expected_gate_durations
    assert tuple(device.metadata.compilation_target_gatesets) == device_info.expected_target_gatesets