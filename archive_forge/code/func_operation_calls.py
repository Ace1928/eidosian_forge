from collections import defaultdict
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane import QubitDevice
from pennylane.measurements import Shots
from pennylane.resource import Resources
from .._version import __version__
def operation_calls(self):
    """Statistics of operation calls"""
    return self._operation_calls