from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate
from ..n_local.n_local import NLocal
@feature_dimension.setter
def feature_dimension(self, feature_dimension: int) -> None:
    """Set the feature dimension.

        Args:
            feature_dimension: The new feature dimension.
        """
    self.num_qubits = feature_dimension