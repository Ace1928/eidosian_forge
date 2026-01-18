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
def findall_operations_with_gate_type(self, gate_type: Type[_TGate]) -> Iterable[Tuple[int, 'cirq.GateOperation', _TGate]]:
    """Find the locations of all gate operations of a given type.

        Args:
            gate_type: The type of gate to find, e.g. XPowGate or
                MeasurementGate.

        Returns:
            An iterator (index, operation, gate)'s for operations with the given
            gate type.
        """
    result = self.findall_operations(lambda operation: isinstance(operation.gate, gate_type))
    for index, op in result:
        gate_op = cast(ops.GateOperation, op)
        yield (index, gate_op, cast(_TGate, gate_op.gate))