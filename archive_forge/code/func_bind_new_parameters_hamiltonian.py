from typing import Sequence, Union
import copy
from functools import singledispatch
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator, Tensor
from ..identity import Identity
from ..qubit import Projector
from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp, Adjoint, Pow, SProd
@bind_new_parameters.register
def bind_new_parameters_hamiltonian(op: qml.Hamiltonian, params: Sequence[TensorLike]):
    new_H = qml.Hamiltonian(params, op.ops)
    if op.grouping_indices is not None:
        new_H.grouping_indices = op.grouping_indices
    return new_H