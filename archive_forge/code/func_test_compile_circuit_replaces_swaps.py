from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compile_circuit_replaces_swaps():
    """Tests that the compiler never sees the SwapPermutationGates from the
    router."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.CNOT(a, c)]), cirq.Moment([cirq.CNOT(b, c)])])
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(model_circuit, device_graph=ccr.gridqubits_to_graph_device(FakeDevice().qubits), compiler=compiler_mock, routing_attempts=1)
    compiler_mock.assert_called_with(compilation_result.circuit)
    assert len(list(compilation_result.circuit.findall_operations_with_gate_type(cirq.ops.SwapPowGate))) > 0
    assert len(list(compilation_result.circuit.findall_operations_with_gate_type(cirq.contrib.acquaintance.SwapPermutationGate))) == 0