from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
def apply_operation_einsum(op: qml.operation.Operator, state, is_state_batched: bool=False):
    """Apply ``Operator`` to ``state`` using ``einsum``. This is more efficent at lower qubit
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
    state_indices = alphabet[:total_indices]
    affected_indices = ''.join((alphabet[i] for i in op.wires))
    new_indices = alphabet[total_indices:total_indices + num_indices]
    new_state_indices = state_indices
    for old, new in zip(affected_indices, new_indices):
        new_state_indices = new_state_indices.replace(old, new)
    einsum_indices = f'...{new_indices}{affected_indices},...{state_indices}->...{new_state_indices}'
    new_mat_shape = [2] * (num_indices * 2)
    dim = 2 ** num_indices
    batch_size = math.get_batch_size(mat, (dim, dim), dim ** 2)
    if batch_size is not None:
        new_mat_shape = [batch_size] + new_mat_shape
        if op.batch_size is None:
            op._batch_size = batch_size
    reshaped_mat = math.reshape(mat, new_mat_shape)
    return math.einsum(einsum_indices, reshaped_mat, state)