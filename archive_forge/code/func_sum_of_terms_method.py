from typing import Callable
from scipy.sparse import csr_matrix
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .apply_operation import apply_operation
def sum_of_terms_method(measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Measure the expecation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    if isinstance(measurementprocess.obs, Sum):
        return sum((measure(ExpectationMP(term), state, is_state_batched=is_state_batched) for term in measurementprocess.obs))
    return sum((c * measure(ExpectationMP(t), state, is_state_batched=is_state_batched) for c, t in zip(*measurementprocess.obs.terms())))