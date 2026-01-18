from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('par,wires,expected_output', [([1, 1], [0, 1], [-1, -1]), ([1], [0], [-1, 1]), ([1], [1], [1, -1])])
def test_basis_state_2_qubit_subset(self, device, tol, par, wires, expected_output):
    """Tests qubit basis state preparation on subsets of qubits"""
    n_wires = 2
    dev = device(n_wires)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array(par), wires=wires)
        return (qml.expval(qml.Z(0)), qml.expval(qml.Z(1)))
    assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))