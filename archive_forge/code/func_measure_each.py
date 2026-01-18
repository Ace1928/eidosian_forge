from typing import Callable, Dict, Iterable, List, overload, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols
from cirq.ops import raw_types, pauli_string
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.pauli_measurement_gate import PauliMeasurementGate
def measure_each(*qubits, key_func: Callable[[raw_types.Qid], str]=str) -> List[raw_types.Operation]:
    """Returns a list of operations individually measuring the given qubits.

    The qubits are measured in the computational basis.

    Args:
        *qubits: The qubits to measure.  These can be passed as separate
            function arguments or as a one-argument iterable of qubits.
        key_func: Determines the key of the measurements of each qubit. Takes
            the qubit and returns the key for that qubit. Defaults to str.

    Returns:
        A list of operations individually measuring the given qubits.
    """
    one_iterable_arg: bool = len(qubits) == 1 and isinstance(qubits[0], Iterable) and (not isinstance(qubits[0], (bytes, str)))
    qubitsequence = qubits[0] if one_iterable_arg else qubits
    return [MeasurementGate(1, key_func(q), qid_shape=(q.dimension,)).on(q) for q in qubitsequence]