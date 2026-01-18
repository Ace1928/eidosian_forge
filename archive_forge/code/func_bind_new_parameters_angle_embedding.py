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
def bind_new_parameters_angle_embedding(op: qml.AngleEmbedding, params: Sequence[TensorLike]):
    rotation = op.hyperparameters['rotation'].basis
    return qml.AngleEmbedding(params[0], wires=op.wires, rotation=rotation)