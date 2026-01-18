from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_generate_model_circuit_without_seed():
    """Test that a model circuit is randomly generated without a seed."""
    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(3, 3)
    assert len(model_circuit) == 3
    assert list(model_circuit.findall_operations_with_gate_type(cirq.MeasurementGate)) == []