from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
@apply_operation.register
def apply_multicontrolledx(op: qml.MultiControlledX, state, is_state_batched: bool=False, debugger=None, **_):
    """Apply MultiControlledX to a state with the default einsum/tensordot choice
    for 8 operation wires or less. Otherwise, apply a custom kernel based on
    composing transpositions, rolling of control axes and the CNOT logic above."""
    if len(op.wires) < 9:
        return _apply_operation_default(op, state, is_state_batched, debugger)
    ctrl_wires = [w + is_state_batched for w in op.control_wires]
    roll_axes = [w for val, w in zip(op.control_values, ctrl_wires) if val is False]
    for ax in roll_axes:
        state = math.roll(state, 1, ax)
    orig_shape = math.shape(state)
    transpose_axes = np.array([w - is_state_batched for w in range(len(orig_shape)) if w - is_state_batched not in op.wires] + [op.wires[-1]] + op.wires[:-1].tolist()) + is_state_batched
    state = math.transpose(state, transpose_axes)
    state = math.reshape(state, (-1, 2, 2 ** (len(op.wires) - 1)))
    state_x = math.roll(state[:, :, -1], 1, 1)[:, :, np.newaxis]
    state = math.concatenate([state[:, :, :-1], state_x], axis=2)
    state = math.transpose(math.reshape(state, orig_shape), np.argsort(transpose_axes))
    for ax in roll_axes:
        state = math.roll(state, 1, ax)
    return state