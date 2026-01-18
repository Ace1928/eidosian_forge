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
def bind_new_parameters_commuting_evolution(op: qml.CommutingEvolution, params: Sequence[TensorLike]):
    new_hamiltonian = bind_new_parameters(op.hyperparameters['hamiltonian'], params[1:])
    freq = op.hyperparameters['frequencies']
    shifts = op.hyperparameters['shifts']
    time = params[0]
    return qml.CommutingEvolution(new_hamiltonian, time, frequencies=freq, shifts=shifts)