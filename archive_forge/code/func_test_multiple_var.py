import pytest
import pennylane as qml
from pennylane import numpy as np  # Import from PennyLane to mirror the standard approach in demos
def test_multiple_var(self, device):
    """Return multiple vars."""
    n_wires = 2
    dev = device(n_wires)
    obs1 = qml.Projector([0], wires=0)
    obs2 = qml.Z(1)
    func = qubit_ansatz

    def circuit(x):
        func(x)
        return (qml.var(obs1), qml.var(obs2))
    qnode = qml.QNode(circuit, dev, diff_method=None)
    res = qnode(0.5)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], (float, np.ndarray))
    assert isinstance(res[1], (float, np.ndarray))