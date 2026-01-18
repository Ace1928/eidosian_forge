from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_grid_device_validate_operations_negative():
    device_info, spec = _create_device_spec_with_horizontal_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    bad_qubit = cirq.GridQubit(10, 10)
    with pytest.raises(ValueError, match='Qubit not on device'):
        device.validate_operation(cirq.X(bad_qubit))
    q00, q10 = (device_info.grid_qubits[0], device_info.grid_qubits[2])
    with pytest.raises(ValueError, match='Qubit pair is not valid'):
        device.validate_operation(cirq.CZ(q00, q10))
    with pytest.raises(ValueError, match='gate which is not supported'):
        device.validate_operation(cirq.H(device_info.grid_qubits[0]))