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
def bind_new_parameters_sprod(op: SProd, params: Sequence[TensorLike]):
    new_scalar = params[0]
    params = params[1:]
    new_base = bind_new_parameters(op.base, params)
    return SProd(new_scalar, new_base)