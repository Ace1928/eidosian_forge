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
def collect_1q_runs(self):
    """Return a set of non-conditional runs of 1q "op" nodes."""

    def filter_fn(node):
        return isinstance(node, DAGOpNode) and len(node.qargs) == 1 and (len(node.cargs) == 0) and isinstance(node.op, Gate) and hasattr(node.op, '__array__') and (getattr(node.op, 'condition', None) is None) and (not node.op.is_parameterized())
    return rx.collect_runs(self._multi_graph, filter_fn)