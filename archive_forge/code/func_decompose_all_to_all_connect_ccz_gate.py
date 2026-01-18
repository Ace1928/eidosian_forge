from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import cirq
def decompose_all_to_all_connect_ccz_gate(ccz_gate: 'cirq.CCZPowGate', qubits: Tuple['cirq.Qid', ...]) -> 'cirq.OP_TREE':
    """Decomposition of all-to-all connected qubits are different from line qubits or grid qubits.

    For example, for qubits in the same ion trap, the decomposition of CCZ gate will be:

    0: ──────────────@──────────────────@───@───p──────@───
                     │                  │   │          │
    1: ───@──────────┼───────@───p──────┼───X───p^-1───X───
          │          │       │          │
    2: ───X───p^-1───X───p───X───p^-1───X───p──────────────

    where p = T**ccz_gate._exponent
    """
    if len(qubits) != 3:
        raise ValueError(f'Expect 3 qubits for CCZ gate, got {len(qubits)} qubits.')
    a, b, c = qubits
    p = cirq.T ** ccz_gate._exponent
    global_phase = 1j ** (2 * ccz_gate.global_shift * ccz_gate._exponent)
    global_phase = complex(global_phase) if cirq.is_parameterized(global_phase) and global_phase.is_complex else global_phase
    global_phase_operation = [cirq.global_phase_operation(global_phase)] if cirq.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0 else []
    return global_phase_operation + [cirq.CNOT(b, c), p(c) ** (-1), cirq.CNOT(a, c), p(c), cirq.CNOT(b, c), p(c) ** (-1), cirq.CNOT(a, c), p(b), p(c), cirq.CNOT(a, b), p(a), p(b) ** (-1), cirq.CNOT(a, b)]