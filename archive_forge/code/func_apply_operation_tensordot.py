from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
def apply_operation_tensordot(op: qml.operation.Operator, state, is_state_batched: bool=False):
    """Apply ``Operator`` to ``state`` using ``math.tensordot``. This is more efficent at higher qubit
    numbers.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        array[complex]: output_state
    """
    mat = op.matrix()
    total_indices = len(state.shape) - is_state_batched
    num_indices = len(op.wires)
    new_mat_shape = [2] * (num_indices * 2)
    dim = 2 ** num_indices
    batch_size = math.get_batch_size(mat, (dim, dim), dim ** 2)
    if (is_mat_batched := (batch_size is not None)):
        new_mat_shape = [batch_size] + new_mat_shape
        if op.batch_size is None:
            op._batch_size = batch_size
    reshaped_mat = math.reshape(mat, new_mat_shape)
    mat_axes = list(range(-num_indices, 0))
    state_axes = [i + is_state_batched for i in op.wires]
    axes = (mat_axes, state_axes)
    tdot = math.tensordot(reshaped_mat, state, axes=axes)
    unused_idxs = [i for i in range(total_indices) if i not in op.wires]
    perm = list(op.wires) + unused_idxs
    if is_mat_batched:
        perm = [0] + [i + 1 for i in perm]
    if is_state_batched:
        perm.insert(num_indices, -1)
    inv_perm = math.argsort(perm)
    return math.transpose(tdot, inv_perm)