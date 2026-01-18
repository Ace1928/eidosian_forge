import functools
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import math
from pennylane import numpy as np
def reshape_state_as_matrix(state, num_wires):
    """Given a non-flat, potentially batched state, flatten it to square matrix or matrices if batched.

    Args:
        state (TensorLike): A state that needs to be reshaped to a square matrix or matrices if batched
        num_wires (int): The number of wires the state represents

    Returns:
        Tensorlike: A reshaped, square state, with an extra batch dimension if necessary
    """
    dim = QUDIT_DIM ** num_wires
    batch_size = math.get_batch_size(state, (QUDIT_DIM,) * (num_wires * 2), dim ** 2)
    shape = (batch_size, dim, dim) if batch_size is not None else (dim, dim)
    return math.reshape(state, shape)