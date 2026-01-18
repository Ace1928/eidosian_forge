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
def next_moment_operating_on(self, qubits: Iterable['cirq.Qid'], start_moment_index: int=0, max_distance: Optional[int]=None) -> Optional[int]:
    """Finds the index of the next moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            start_moment_index: The starting point of the search.
            max_distance: The number of moments (starting from the start index
                and moving forward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            earliest matching moment.

        Raises:
          ValueError: negative max_distance.
        """
    max_circuit_distance = len(self.moments) - start_moment_index
    if max_distance is None:
        max_distance = max_circuit_distance
    elif max_distance < 0:
        raise ValueError(f'Negative max_distance: {max_distance}')
    else:
        max_distance = min(max_distance, max_circuit_distance)
    return self._first_moment_operating_on(qubits, range(start_moment_index, start_moment_index + max_distance))