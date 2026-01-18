from typing import Callable, Dict, Iterable, List, overload, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols
from cirq.ops import raw_types, pauli_string
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.pauli_measurement_gate import PauliMeasurementGate
def measure_single_paulistring(pauli_observable: pauli_string.PauliString, key: Optional[Union[str, 'cirq.MeasurementKey']]=None) -> raw_types.Operation:
    """Returns a single PauliMeasurementGate which measures the pauli observable

    Args:
        pauli_observable: The `cirq.PauliString` observable to measure.
        key: Optional `str` or `cirq.MeasurementKey` that gate should use.
            If none provided, it defaults to a comma-separated list of
            `str(qubit)` for each of the target qubits.

    Returns:
        An operation measuring the pauli observable.

    Raises:
        ValueError: if the observable is not an instance of PauliString or if the coefficient
            is not +1 or -1.
    """
    if not isinstance(pauli_observable, pauli_string.PauliString):
        raise ValueError(f'Pauli observable {pauli_observable} should be an instance of cirq.PauliString.')
    if abs(pauli_observable.coefficient) != 1:
        raise ValueError(f'Pauli observable {pauli_observable} must have a coefficient of +1 or -1.')
    if key is None:
        key = _default_measurement_key(pauli_observable)
    return PauliMeasurementGate(pauli_observable.dense(list(pauli_observable.keys())), key).on(*pauli_observable.keys())