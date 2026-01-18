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
def insert_into_range(self, operations: 'cirq.OP_TREE', start: int, end: int) -> int:
    """Writes operations inline into an area of the circuit.

        Args:
            start: The start of the range (inclusive) to write the
                given operations into.
            end: The end of the range (exclusive) to write the given
                operations into. If there are still operations remaining,
                new moments are created to fit them.
            operations: An operation or tree of operations to insert.

        Returns:
            An insertion index that will place operations after the operations
            that were inserted by this method.

        Raises:
            IndexError: Bad inline_start and/or inline_end.
        """
    if not 0 <= start <= end <= len(self):
        raise IndexError(f'Bad insert indices: [{start}, {end})')
    flat_ops = list(ops.flatten_to_ops(operations))
    i = start
    op_index = 0
    while op_index < len(flat_ops):
        op = flat_ops[op_index]
        while i < end and self._moments[i].operates_on(op.qubits):
            i += 1
        if i >= end:
            break
        self._moments[i] = self._moments[i].with_operation(op)
        op_index += 1
    self._mutated()
    if op_index >= len(flat_ops):
        return end
    return self.insert(end, flat_ops[op_index:])