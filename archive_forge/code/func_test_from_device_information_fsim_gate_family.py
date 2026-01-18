from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def test_from_device_information_fsim_gate_family():
    """Verifies that FSimGateFamilies are recognized correctly."""
    gateset = cirq.Gateset(cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]))
    device = grid_device.GridDevice._from_device_information(qubit_pairs=[(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))], gateset=gateset)
    assert gateset.gates.issubset(device.metadata.gateset.gates)