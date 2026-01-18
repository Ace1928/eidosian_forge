import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def circuit1(param):
    """First Pauli subcircuit"""
    qml.RX(param, wires=0)
    qml.RY(param, wires=0)
    return qml.expval(qml.X(0))