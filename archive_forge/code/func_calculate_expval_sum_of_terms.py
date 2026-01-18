from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def calculate_expval_sum_of_terms(measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Measure the expectation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: the expectation value of the sum of Hamiltonian observable wrt the state.
    """
    if isinstance(measurementprocess.obs, Sum):
        return sum((measure(ExpectationMP(term), state, is_state_batched=is_state_batched) for term in measurementprocess.obs))
    return sum((c * measure(ExpectationMP(t), state, is_state_batched=is_state_batched) for c, t in zip(*measurementprocess.obs.terms())))