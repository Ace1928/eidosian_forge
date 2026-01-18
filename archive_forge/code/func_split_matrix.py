from functools import lru_cache, reduce
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
def split_matrix(theta):
    """Compute the real and imaginary parts of the special unitary matrix."""
    mat = self.compute_matrix(theta, num_wires)
    return (qml.math.real(mat), qml.math.imag(mat))