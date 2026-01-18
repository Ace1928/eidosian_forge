from typing import Callable, Dict, Iterable, List, overload, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols
from cirq.ops import raw_types, pauli_string
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.pauli_measurement_gate import PauliMeasurementGate
def measure_paulistring_terms(pauli_basis: pauli_string.PauliString, key_func: Callable[[raw_types.Qid], str]=str) -> List[raw_types.Operation]:
    """Returns a list of operations individually measuring qubits in the pauli basis.

    Args:
        pauli_basis: The `cirq.PauliString` basis in which each qubit should
            be measured.
        key_func: Determines the key of the measurements of each qubit. Takes
            the qubit and returns the key for that qubit. Defaults to str.

    Returns:
        A list of operations individually measuring the given qubits in the
        specified pauli basis.

    Raises:
        ValueError: if `pauli_basis` is not an instance of `cirq.PauliString`.
    """
    if not isinstance(pauli_basis, pauli_string.PauliString):
        raise ValueError(f'Pauli observable {pauli_basis} should be an instance of cirq.PauliString.')
    return [PauliMeasurementGate([pauli_basis[q]], key=key_func(q)).on(q) for q in pauli_basis]