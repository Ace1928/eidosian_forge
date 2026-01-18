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
def findall_operations_between(self, start_frontier: Dict['cirq.Qid', int], end_frontier: Dict['cirq.Qid', int], omit_crossing_operations: bool=False) -> List[Tuple[int, 'cirq.Operation']]:
    """Finds operations between the two given frontiers.

        If a qubit is in `start_frontier` but not `end_frontier`, its end index
        defaults to the end of the circuit. If a qubit is in `end_frontier` but
        not `start_frontier`, its start index defaults to the start of the
        circuit. Operations on qubits not mentioned in either frontier are not
        included in the results.

        Args:
            start_frontier: Just before where to start searching for operations,
                for each qubit of interest. Start frontier indices are
                inclusive.
            end_frontier: Just before where to stop searching for operations,
                for each qubit of interest. End frontier indices are exclusive.
            omit_crossing_operations: Determines whether or not operations that
                cross from a location between the two frontiers to a location
                outside the two frontiers are included or excluded. (Operations
                completely inside are always included, and operations completely
                outside are always excluded.)

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the two frontiers. The first item of each tuple is the index of the
            moment containing the operation, and the second item is the
            operation itself. The list is sorted so that the moment index
            increases monotonically.
        """
    result = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)
    involved_qubits = set(start_frontier.keys()) | set(end_frontier.keys())
    for q in sorted(involved_qubits):
        for i in range(start_frontier.get(q, 0), end_frontier.get(q, len(self))):
            op = self.operation_at(q, i)
            if op is None:
                continue
            if omit_crossing_operations and (not involved_qubits.issuperset(op.qubits)):
                continue
            result.enqueue(i, op)
    return list(result)