from typing import Tuple, cast
from cirq import circuits, ops, protocols, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def continue_condition(op: ops.Operation, current_string: ops.PauliStringPhasor, is_first: bool) -> int:
    if isinstance(op.gate, ops.SingleQubitCliffordGate):
        return CONTINUE if len(current_string.pauli_string) != 1 else STOP
    if isinstance(op.gate, ops.CZPowGate):
        return STOP if stop_at_cz else CONTINUE
    if isinstance(op, ops.PauliStringPhasor) and len(op.qubits) == 1 and (op.pauli_string[op.qubits[0]] == current_string.pauli_string[op.qubits[0]]):
        return SKIP
    return STOP