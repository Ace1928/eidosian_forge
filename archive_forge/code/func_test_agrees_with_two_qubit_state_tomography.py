from typing import Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('circuit', (cirq.Circuit(cirq.CNOT(Q0, Q1) ** 0.3), cirq.Circuit(cirq.H(Q0), cirq.CNOT(Q0, Q1)), cirq.Circuit(cirq.X(Q0) ** 0.25, cirq.ISWAP(Q0, Q1))))
def test_agrees_with_two_qubit_state_tomography(circuit):
    qubits = (Q0, Q1)
    sim = cirq.Simulator(seed=87539319)
    tomography_result = cirq.experiments.state_tomography(sim, qubits, circuit, repetitions=5000)
    actual_rho = tomography_result.data
    two_qubit_tomography_result = cirq.experiments.two_qubit_state_tomography(sim, qubits[0], qubits[1], circuit, repetitions=5000)
    expected_rho = two_qubit_tomography_result.data
    error_rho = actual_rho - expected_rho
    assert np.linalg.norm(error_rho) < 0.06
    assert np.max(np.abs(error_rho)) < 0.05