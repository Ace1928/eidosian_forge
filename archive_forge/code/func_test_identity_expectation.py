import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_identity_expectation(self, device, tol):
    """Test that identity expectation value (i.e. the trace) is 1."""
    n_wires = 2
    dev = device(n_wires)
    theta = 0.432
    phi = 0.123

    @qml.qnode(dev)
    def circuit():
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return (qml.expval(qml.Identity(wires=0)), qml.expval(qml.Identity(wires=1)))
    res = circuit()
    assert np.allclose(res, np.array([1, 1]), atol=tol(dev.shots))