from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_identity(op: qml.Identity, state, is_state_batched: bool=False, debugger=None, **_):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state