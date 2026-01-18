from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_from_device_information_empty():
    device = grid_device.GridDevice._from_device_information(qubit_pairs=[], gateset=cirq.Gateset(), gate_durations=None)
    assert len(device.metadata.qubit_set) == 0
    assert len(device.metadata.qubit_pairs) == 0
    assert device.metadata.gateset == cirq.Gateset()
    assert device.metadata.gate_durations is None