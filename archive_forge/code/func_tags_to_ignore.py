from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
@property
def tags_to_ignore(self) -> FrozenSet[Hashable]:
    return self._tags_to_ignore