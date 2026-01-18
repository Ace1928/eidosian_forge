from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
def skip_diagonalizing(obs):
    return isinstance(obs, qml.Hamiltonian) or (isinstance(obs, qml.ops.Sum) and obs._pauli_rep is not None)