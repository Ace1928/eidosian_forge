import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def prev_moment_operating_on(self, qubits: Sequence['cirq.Qid'], end_moment_index: Optional[int]=None, max_distance: Optional[int]=None) -> Optional[int]:
    """Finds the index of the previous moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            end_moment_index: The moment index just after the starting point of
                the reverse search. Defaults to the length of the list of
                moments.
            max_distance: The number of moments (starting just before from the
                end index and moving backward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            latest matching moment.

        Raises:
            ValueError: negative max_distance.
        """
    if end_moment_index is None:
        end_moment_index = len(self.moments)
    if max_distance is None:
        max_distance = len(self.moments)
    elif max_distance < 0:
        raise ValueError(f'Negative max_distance: {max_distance}')
    else:
        max_distance = min(end_moment_index, max_distance)
    if end_moment_index > len(self.moments):
        d = end_moment_index - len(self.moments)
        end_moment_index -= d
        max_distance -= d
    if max_distance <= 0:
        return None
    return self._first_moment_operating_on(qubits, (end_moment_index - k - 1 for k in range(max_distance)))