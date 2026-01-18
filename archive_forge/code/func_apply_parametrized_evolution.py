from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_parametrized_evolution(op: qml.pulse.ParametrizedEvolution, state, is_state_batched: bool=False, debugger=None, **_):
    """Apply ParametrizedEvolution by evolving the state rather than the operator matrix
    if we are operating on more than half of the subsystem"""
    num_wires = len(qml.math.shape(state)) - is_state_batched
    state = qml.math.cast(state, complex)
    if 2 * len(op.wires) <= num_wires or op.hyperparameters['complementary'] or (is_state_batched and op.hyperparameters['return_intermediate']):
        return _apply_operation_default(op, state, is_state_batched, debugger)
    return _evolve_state_vector_under_parametrized_evolution(op, state, num_wires, is_state_batched)