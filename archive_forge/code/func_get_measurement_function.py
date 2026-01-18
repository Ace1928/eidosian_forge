from typing import Callable
from scipy.sparse import csr_matrix
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .apply_operation import apply_operation
def get_measurement_function(measurementprocess: MeasurementProcess, state: TensorLike) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        Callable: function that returns the measurement result
    """
    if isinstance(measurementprocess, StateMeasurement):
        if isinstance(measurementprocess.mv, MeasurementValue):
            return state_diagonalizing_gates
        if isinstance(measurementprocess, ExpectationMP):
            if measurementprocess.obs.name == 'SparseHamiltonian':
                return csr_dot_products
            if measurementprocess.obs.name == 'Hermitian':
                return full_dot_products
            backprop_mode = math.get_interface(state, *measurementprocess.obs.data) != 'numpy'
            if isinstance(measurementprocess.obs, Hamiltonian):
                return sum_of_terms_method if backprop_mode else csr_dot_products
            if isinstance(measurementprocess.obs, Sum):
                if backprop_mode:
                    return sum_of_terms_method
                if measurementprocess.obs.has_overlapping_wires and len(measurementprocess.obs.wires) > 7:
                    return csr_dot_products
        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            return state_diagonalizing_gates
    raise NotImplementedError