from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def calculate_probability(measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Find the probability of measuring states.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: the probability of the state being in each measurable state.
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state, is_state_batched=is_state_batched)
    num_state_wires = get_num_wires(state, is_state_batched)
    reshaped_state = reshape_state_as_matrix(state, num_state_wires)
    if is_state_batched:
        probs = math.real(math.stack([math.diagonal(dm) for dm in reshaped_state]))
    else:
        probs = math.real(math.diagonal(reshaped_state))
    probs = math.where(probs < 0, 0, probs)
    if (mp_wires := measurementprocess.wires):
        expanded_shape = [QUDIT_DIM] * num_state_wires
        new_shape = [QUDIT_DIM ** len(mp_wires)]
        if is_state_batched:
            batch_size = probs.shape[0]
            expanded_shape.insert(0, batch_size)
            new_shape.insert(0, batch_size)
        wires_to_trace = tuple((x + is_state_batched for x in range(num_state_wires) if x not in mp_wires))
        expanded_probs = math.reshape(probs, expanded_shape)
        summed_probs = math.sum(expanded_probs, axis=wires_to_trace)
        return math.reshape(summed_probs, new_shape)
    return probs