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
def count_ops_longest_path(self):
    """Count the occurrences of operation names on the longest path.

        Returns a dictionary of counts keyed on the operation name.
        """
    op_dict = {}
    path = self.longest_path()
    path = path[1:-1]
    for node in path:
        name = node.op.name
        if name not in op_dict:
            op_dict[name] = 1
        else:
            op_dict[name] += 1
    return op_dict