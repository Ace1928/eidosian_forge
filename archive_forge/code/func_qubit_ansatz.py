import pytest
import pennylane as qml
from pennylane import numpy as np  # Import from PennyLane to mirror the standard approach in demos
def qubit_ansatz(x):
    """Qfunc ansatz"""
    qml.Hadamard(wires=[0])
    qml.CRX(x, wires=[0, 1])