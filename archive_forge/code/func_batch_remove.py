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
def batch_remove(self, removals: Iterable[Tuple[int, 'cirq.Operation']]) -> None:
    """Removes several operations from a circuit.

        Args:
            removals: A sequence of (moment_index, operation) tuples indicating
                operations to delete from the moments that are present. All
                listed operations must actually be present or the edit will
                fail (without making any changes to the circuit).

        Raises:
            ValueError: One of the operations to delete wasn't present to start with.
            IndexError: Deleted from a moment that doesn't exist.
        """
    copy = self.copy()
    for i, op in removals:
        if op not in copy._moments[i].operations:
            raise ValueError(f"Can't remove {op} @ {i} because it doesn't exist.")
        copy._moments[i] = Moment((old_op for old_op in copy._moments[i].operations if op != old_op))
    self._moments = copy._moments
    self._mutated()