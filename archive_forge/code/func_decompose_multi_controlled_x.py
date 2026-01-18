from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def decompose_multi_controlled_x(controls: List['cirq.Qid'], target: 'cirq.Qid', free_qubits: List['cirq.Qid']) -> List['cirq.Operation']:
    """Implements action of multi-controlled Pauli X gate.

    Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
    gates.
    If `free_qubits` has at least 1 element, result has lengts
    O(len(controls)).

    Args:
        controls - control qubits.
        targets - target qubits.
        free_qubits - qubits which are neither controlled nor target. Can be
            modified by algorithm, but will end up in their initial state.
    """
    m = len(controls)
    if m == 0:
        return [ops.X.on(target)]
    elif m == 1:
        return [ops.CNOT.on(controls[0], target)]
    elif m == 2:
        return [ops.CCNOT.on(controls[0], controls[1], target)]
    m = len(controls)
    n = m + 1 + len(free_qubits)
    if n >= 2 * m - 1 and m >= 3:
        gates1 = [_ccnot_congruent(controls[m - 2 - i], free_qubits[m - 4 - i], free_qubits[m - 3 - i]) for i in range(m - 3)]
        gates2 = _ccnot_congruent(controls[0], controls[1], free_qubits[0])
        gates3 = _flatten(gates1) + gates2 + _flatten(gates1[::-1])
        first_ccnot = ops.CCNOT(controls[m - 1], free_qubits[m - 3], target)
        return [first_ccnot, *gates3, first_ccnot, *gates3]
    elif len(free_qubits) >= 1:
        m1 = n // 2
        free1 = controls[m1:] + [target] + free_qubits[1:]
        ctrl1 = controls[:m1]
        part1 = decompose_multi_controlled_x(ctrl1, free_qubits[0], free1)
        free2 = controls[:m1] + free_qubits[1:]
        ctrl2 = controls[m1:] + [free_qubits[0]]
        part2 = decompose_multi_controlled_x(ctrl2, target, free2)
        return [*part1, *part2, *part1, *part2]
    else:
        return decompose_multi_controlled_rotation(unitary(ops.X), controls, target)