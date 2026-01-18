import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def operation_at(self, qubit: raw_types.Qid) -> Optional['cirq.Operation']:
    """Returns the operation on a certain qubit for the moment.

        Args:
            qubit: The qubit on which the returned Operation operates
                on.

        Returns:
            The operation that operates on the qubit for that moment.
        """
    if self.operates_on([qubit]):
        return self.__getitem__(qubit)
    return None