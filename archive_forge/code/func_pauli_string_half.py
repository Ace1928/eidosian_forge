from typing import Tuple
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def pauli_string_half(circuit: circuits.Circuit) -> circuits.Circuit:
    """Return only the non-Clifford part of a circuit.  See
    convert_and_separate_circuit().

    Args:
        circuit: A Circuit with the gate set {SingleQubitCliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        A Circuit with only PauliStringPhasor operations.
    """
    return circuits.Circuit(_pull_non_clifford_before(circuit), strategy=circuits.InsertStrategy.EARLIEST)