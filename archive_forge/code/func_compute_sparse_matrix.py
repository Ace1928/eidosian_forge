import copy
from typing import Union
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply
from .symbolicop import ScalarSymbolicOp
@staticmethod
def compute_sparse_matrix(*params, base=None, z=0):
    if isinstance(z, int):
        base_matrix = base.compute_sparse_matrix(*params, **base.hyperparameters)
        return base_matrix ** z
    raise SparseMatrixUndefinedError