from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def calculate_variance(measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Find variance of observable.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state.
        state (TensorLike): state to apply the measurement to.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: the variance of the observable wrt the state.
    """
    probs = calculate_probability(measurementprocess, state, is_state_batched)
    eigvals = math.asarray(measurementprocess.eigvals(), dtype='float64')
    return math.dot(probs, eigvals ** 2) - math.dot(probs, eigvals) ** 2