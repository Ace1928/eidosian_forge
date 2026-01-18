from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
@pytest.mark.parametrize('spec, error_match', [(_create_device_spec_duplicate_qubit(), 'Invalid DeviceSpecification: .*duplicate qubit'), (_create_device_spec_invalid_qubit_name(), 'Invalid DeviceSpecification: .*not in the GridQubit form'), (_create_device_spec_invalid_qubit_in_qubit_pair(), 'Invalid DeviceSpecification: .*which is not in valid_qubits'), (_create_device_spec_qubit_pair_self_loops(), 'Invalid DeviceSpecification: .*contains repeated qubits'), (_create_device_spec_unexpected_asymmetric_target(), 'Invalid DeviceSpecification: .*cannot be ASYMMETRIC')])
def test_grid_device_invalid_device_specification(spec, error_match):
    with pytest.raises(ValueError, match=error_match):
        cirq_google.GridDevice.from_proto(spec)