from typing import Tuple
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def regular_half(circuit: circuits.Circuit) -> circuits.Circuit:
    """Return only the Clifford part of a circuit.  See
    convert_and_separate_circuit().

    Args:
        circuit: A Circuit with the gate set {SingleQubitCliffordGate,
            PauliInteractionGate, PauliStringPhasor}.

    Returns:
        A Circuit with SingleQubitCliffordGate and PauliInteractionGate gates.
        It also contains MeasurementGates if the given
        circuit contains measurements.
    """
    return circuits.Circuit((circuits.Moment((op for op in moment.operations if not isinstance(op, ops.PauliStringPhasor))) for moment in circuit))