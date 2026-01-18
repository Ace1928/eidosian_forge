from typing import Tuple
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def convert_and_separate_circuit(circuit: circuits.Circuit, leave_cliffords: bool=True, atol: float=1e-08) -> Tuple[circuits.Circuit, circuits.Circuit]:
    """Converts a circuit into two, one made of PauliStringPhasor and the other Clifford gates.

    Args:
        circuit: Any Circuit that cirq.google.optimized_for_xmon() supports.
            All gates should either provide a decomposition or have a known one
            or two qubit unitary matrix.
        leave_cliffords: If set, single qubit rotations in the Clifford group
                are not converted to SingleQubitCliffordGates.
        atol: The absolute tolerance for the conversion.

    Returns:
        (circuit_left, circuit_right)

        circuit_left contains only PauliStringPhasor operations.

        circuit_right is a Clifford circuit which contains only
        SingleQubitCliffordGate and PauliInteractionGate gates.
        It also contains MeasurementGates if the
        given circuit contains measurements.

    """
    single_qubit_target = CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS_AND_CLIFFORDS if leave_cliffords else CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS
    circuit = transformers.optimize_for_target_gateset(circuit, gateset=CliffordTargetGateset(atol=atol, single_qubit_target=single_qubit_target))
    return (pauli_string_half(circuit), regular_half(circuit))