from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def decompose_to_target_gateset(self, op: 'cirq.Operation', _) -> DecomposeResult:
    return op if cirq.num_qubits(op) == 2 and cirq.has_unitary(op) else NotImplemented