import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_multi_mode_hermitian_expectation(self, device, tol):
    """Test that arbitrary multi-mode Hermitian expectation values are correct"""
    n_wires = 2
    dev = device(n_wires)
    if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
        pytest.skip('Skipped because device does not support the Hermitian observable.')
    theta = 0.432
    phi = 0.123
    A_ = np.array([[-6, 2 + 1j, -3, -5 + 2j], [2 - 1j, 0, 2 - 1j, -5 + 4j], [-3, 2 + 1j, 0, -4 + 3j], [-5 - 2j, -5 - 4j, -4 - 3j, -6]])

    @qml.qnode(dev)
    def circuit():
        qml.RY(theta, wires=[0])
        qml.RY(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Hermitian(A_, wires=[0, 1]))
    res = circuit()
    expected = 0.5 * (6 * np.cos(theta) * np.sin(phi) - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3) - 2 * np.sin(phi) - 6 * np.cos(phi) - 6)
    assert np.allclose(res, expected, atol=tol(dev.shots))