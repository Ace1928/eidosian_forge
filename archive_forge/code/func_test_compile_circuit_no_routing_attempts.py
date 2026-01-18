from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compile_circuit_no_routing_attempts():
    """Tests that setting no routing attempts throws an error."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)])])
    with pytest.raises(AssertionError) as e:
        cirq.contrib.quantum_volume.compile_circuit(model_circuit, device_graph=ccr.gridqubits_to_graph_device(FakeDevice().qubits), routing_attempts=0)
    assert e.match('Unable to get routing for circuit')