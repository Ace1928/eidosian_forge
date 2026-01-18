from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compile_circuit_router():
    """Tests that the given router is used."""
    router_mock = MagicMock()
    cirq.contrib.quantum_volume.compile_circuit(cirq.Circuit(), device_graph=ccr.gridqubits_to_graph_device(FakeDevice().qubits), router=router_mock, routing_attempts=1)
    router_mock.assert_called()