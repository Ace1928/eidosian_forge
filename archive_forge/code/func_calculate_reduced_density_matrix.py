from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def calculate_reduced_density_matrix(measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Get the state or reduced density matrix.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: state or reduced density matrix.
    """
    wires = measurementprocess.wires
    if not wires:
        return reshape_state_as_matrix(state, get_num_wires(state, is_state_batched))
    num_obs_wires = len(wires)
    num_state_wires = get_num_wires(state, is_state_batched)
    state_wire_indices_list = list(alphabet[:num_state_wires] * 2)
    final_state_wire_indices_list = [''] * (2 * num_obs_wires)
    for i, wire in enumerate(wires):
        col_index = wire + num_state_wires
        state_wire_indices_list[col_index] = alphabet[col_index]
        final_state_wire_indices_list[i] = alphabet[wire]
        final_state_wire_indices_list[i + num_obs_wires] = alphabet[col_index]
    state_wire_indices = ''.join(state_wire_indices_list)
    final_state_wire_indices = ''.join(final_state_wire_indices_list)
    state = math.einsum(f'...{state_wire_indices}->...{final_state_wire_indices}', state)
    return reshape_state_as_matrix(state, len(wires))