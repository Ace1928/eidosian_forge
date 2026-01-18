from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_cnot(op: qml.CNOT, state, is_state_batched: bool=False, debugger=None, **_):
    """Apply cnot gate to state."""
    target_axes = (op.wires[1] - 1 if op.wires[1] > op.wires[0] else op.wires[1]) + is_state_batched
    control_axes = op.wires[0] + is_state_batched
    n_dim = math.ndim(state)
    if n_dim >= 9 and math.get_interface(state) == 'tensorflow':
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)
    sl_0 = _get_slice(0, control_axes, n_dim)
    sl_1 = _get_slice(1, control_axes, n_dim)
    state_x = math.roll(state[sl_1], 1, target_axes)
    return math.stack([state[sl_0], state_x], axis=control_axes)