from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_grid_device_str_grid_qubits():
    spec = _create_device_spec_with_all_couplings()
    device = cirq_google.GridDevice.from_proto(spec)
    assert str(device) == '(0, 0)───(0, 1)\n│        │\n│        │\n(1, 0)───(1, 1)\n│        │\n│        │\n(2, 0)───(2, 1)\n│        │\n│        │\n(3, 0)───(3, 1)\n│        │\n│        │\n(4, 0)───(4, 1)'