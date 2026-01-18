from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
@property
def gates(self) -> FrozenSet[GateFamily]:
    return self._gates