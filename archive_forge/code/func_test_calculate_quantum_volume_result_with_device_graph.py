from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_calculate_quantum_volume_result_with_device_graph():
    """Test that running the main loop routes the circuit onto the given device
    graph"""
    device_qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(3)]
    results = cirq.contrib.quantum_volume.calculate_quantum_volume(num_qubits=3, depth=3, num_circuits=1, device_graph=ccr.gridqubits_to_graph_device(device_qubits), samplers=[cirq.Simulator()], routing_attempts=2, random_state=1)
    assert len(results) == 1
    assert ccr.ops_are_consistent_with_device_graph(results[0].compiled_circuit.all_operations(), ccr.get_grid_device_graph(2, 3))