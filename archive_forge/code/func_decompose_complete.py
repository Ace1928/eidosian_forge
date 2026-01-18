import enum
import itertools
from typing import Dict, Sequence, Tuple, Union, TYPE_CHECKING
from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate, SwapPermutationGate
def decompose_complete(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
    swap_gate = SwapPermutationGate(self.swap_gate)
    if self.part_size == 1:
        yield acquaint(*qubits)
        return
    for k in range(-self.part_size + 1, self.part_size - 1):
        for x in range(abs(k), 2 * self.part_size - abs(k), 2):
            yield acquaint(*qubits[x:x + 2])
            yield swap_gate(*qubits[x:x + 2])
    yield acquaint(qubits[self.part_size - 1], qubits[self.part_size])
    for k in reversed(range(-self.part_size + 1, self.part_size - 1)):
        for x in range(abs(k), 2 * self.part_size - abs(k), 2):
            yield acquaint(*qubits[x:x + 2])
            yield swap_gate(*qubits[x:x + 2])