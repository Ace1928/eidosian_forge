from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate
from ..n_local.n_local import NLocal
def self_product(x: np.ndarray) -> float:
    """
    Define a function map from R^n to R.

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.pi - x)
    return coeff