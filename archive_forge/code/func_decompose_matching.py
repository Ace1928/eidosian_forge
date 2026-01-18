import enum
import itertools
from typing import Dict, Sequence, Tuple, Union, TYPE_CHECKING
from cirq import ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.permutation import PermutationGate, SwapPermutationGate
def decompose_matching(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
    swap_gate = SwapPermutationGate(self.swap_gate)
    for k in range(-self.part_size + 1, self.part_size):
        for x in range(abs(k), 2 * self.part_size - abs(k), 2):
            if (x + 1) % self.part_size:
                yield swap_gate(*qubits[x:x + 2])
            else:
                yield acquaint(*qubits[x:x + 2])