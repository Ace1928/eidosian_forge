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
def bind_new_parameters_tensor(op: Tensor, params: Sequence[TensorLike]):
    new_obs = []
    for obs in op.obs:
        sub_params = params[:obs.num_params]
        params = params[obs.num_params:]
        new_obs.append(bind_new_parameters(obs, sub_params))
    return Tensor(*new_obs)