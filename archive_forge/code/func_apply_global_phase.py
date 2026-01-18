from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_global_phase(op: qml.GlobalPhase, state, is_state_batched: bool=False, debugger=None, **_):
    """Applies a :class:`~.GlobalPhase` operation by multiplying the state by ``exp(1j * op.data[0])``"""
    return qml.math.exp(-1j * qml.math.cast(op.data[0], complex)) * state