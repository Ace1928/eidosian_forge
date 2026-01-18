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
def bind_new_parameters_fermionic_double_excitation(op: qml.FermionicDoubleExcitation, params: Sequence[TensorLike]):
    wires1 = op.hyperparameters['wires1']
    wires2 = op.hyperparameters['wires2']
    return qml.FermionicDoubleExcitation(params[0], wires1=wires1, wires2=wires2)