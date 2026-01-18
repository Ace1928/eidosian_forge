from typing import Callable
from scipy.sparse import csr_matrix
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .apply_operation import apply_operation
def full_dot_products(measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Measure the expectation value of an observable using the dot product between full matrix
    representations.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    ket = apply_operation(measurementprocess.obs, state, is_state_batched=is_state_batched)
    dot_product = math.sum(math.conj(state) * ket, axis=tuple(range(int(is_state_batched), math.ndim(state))))
    return math.real(dot_product)