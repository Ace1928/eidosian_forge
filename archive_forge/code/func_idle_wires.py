from node A to node B means that the (qu)bit passes from the output of A
from collections import OrderedDict, defaultdict, deque, namedtuple
import copy
import math
from typing import Dict, Generator, Any, List
import numpy as np
import rustworkx as rx
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources, node_resources, CONTROL_FLOW_OP_NAMES
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit.bit import Bit
def idle_wires(self, ignore=None):
    """Return idle wires.

        Args:
            ignore (list(str)): List of node names to ignore. Default: []

        Yields:
            Bit: Bit in idle wire.

        Raises:
            DAGCircuitError: If the DAG is invalid
        """
    if ignore is None:
        ignore = set()
    ignore_set = set(ignore)
    for wire in self._wires:
        if not ignore:
            if self._is_wire_idle(wire):
                yield wire
        else:
            for node in self.nodes_on_wire(wire, only_ops=True):
                if node.op.name not in ignore_set:
                    break
            else:
                yield wire