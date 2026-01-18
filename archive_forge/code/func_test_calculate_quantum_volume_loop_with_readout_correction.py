from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_calculate_quantum_volume_loop_with_readout_correction():
    """Test that calculate_quantum_volume is able to run without erring with
    readout error correction."""
    cirq.contrib.quantum_volume.calculate_quantum_volume(num_qubits=4, depth=4, num_circuits=1, routing_attempts=2, random_state=1, device_graph=ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3)), samplers=[cirq.Simulator()], add_readout_error_correction=True)