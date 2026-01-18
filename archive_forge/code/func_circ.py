import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def circ(wire_labels):
    sub_routine(wire_labels)
    return qml.var(qml.X(wire_labels[0]) @ qml.Y(wire_labels[1]) @ qml.Z(wire_labels[2]))