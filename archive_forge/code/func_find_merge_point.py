from typing import Tuple, cast
from cirq import circuits, ops, protocols, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def find_merge_point(start_i: int, string_op: ops.PauliStringPhasor, stop_at_cz: bool) -> Tuple[int, ops.PauliStringPhasor, int]:
    STOP = 0
    CONTINUE = 1
    SKIP = 2

    def continue_condition(op: ops.Operation, current_string: ops.PauliStringPhasor, is_first: bool) -> int:
        if isinstance(op.gate, ops.SingleQubitCliffordGate):
            return CONTINUE if len(current_string.pauli_string) != 1 else STOP
        if isinstance(op.gate, ops.CZPowGate):
            return STOP if stop_at_cz else CONTINUE
        if isinstance(op, ops.PauliStringPhasor) and len(op.qubits) == 1 and (op.pauli_string[op.qubits[0]] == current_string.pauli_string[op.qubits[0]]):
            return SKIP
        return STOP
    modified_op = string_op
    furthest_op = string_op
    furthest_i = start_i + 1
    num_passed_over = 0
    for i in range(start_i + 1, len(all_ops)):
        op = all_ops[i]
        if not set(op.qubits) & set(modified_op.qubits):
            continue
        cont_cond = continue_condition(op, modified_op, i == start_i + 1)
        if cont_cond == STOP:
            if len(modified_op.pauli_string) == 1:
                furthest_op = modified_op
                furthest_i = i
            break
        if cont_cond == CONTINUE:
            modified_op = modified_op.pass_operations_over([op], after_to_before=True)
        num_passed_over += 1
        if len(modified_op.pauli_string) == 1:
            furthest_op = modified_op
            furthest_i = i + 1
    return (furthest_i, furthest_op, num_passed_over)