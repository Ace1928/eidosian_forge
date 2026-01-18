from typing import Sequence, Union
import copy
from functools import singledispatch
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator, Tensor
from ..identity import Identity
from ..qubit import Projector
from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp, Adjoint, Pow, SProd
@bind_new_parameters.register(qml.CY)
@bind_new_parameters.register(qml.CZ)
@bind_new_parameters.register(qml.CH)
@bind_new_parameters.register(qml.CCZ)
@bind_new_parameters.register(qml.CSWAP)
@bind_new_parameters.register(qml.CNOT)
@bind_new_parameters.register(qml.Toffoli)
@bind_new_parameters.register(qml.MultiControlledX)
def bind_new_parameters_copy(op, params: Sequence[TensorLike]):
    return copy.copy(op)