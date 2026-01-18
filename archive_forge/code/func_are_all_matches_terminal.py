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
def are_all_matches_terminal(self, predicate: Callable[['cirq.Operation'], bool]) -> bool:
    """Check whether all of the ops that satisfy a predicate are terminal.

        This method will transparently descend into any CircuitOperations this
        circuit contains; as a result, it will misbehave if the predicate
        refers to CircuitOperations. See the tests for an example of this.

        Args:
            predicate: A predicate on ops.Operations which is being checked.

        Returns:
            Whether or not all `Operation` s in a circuit that satisfy the
            given predicate are terminal. Also checks within any CircuitGates
            the circuit may contain.
        """
    from cirq.circuits import CircuitOperation
    if not all((self.next_moment_operating_on(op.qubits, i + 1) is None for i, op in self.findall_operations(predicate) if not isinstance(op.untagged, CircuitOperation))):
        return False
    for i, moment in enumerate(self.moments):
        for op in moment.operations:
            circuit = getattr(op.untagged, 'circuit', None)
            if circuit is None:
                continue
            if not circuit.are_all_matches_terminal(predicate):
                return False
            if i < len(self.moments) - 1 and (not all((self.next_moment_operating_on(op.qubits, i + 1) is None for _, op in circuit.findall_operations(predicate)))):
                return False
    return True