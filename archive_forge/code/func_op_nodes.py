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
def op_nodes(self, op=None, include_directives=True):
    """Get the list of "op" nodes in the dag.

        Args:
            op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to
                return. If None, return all op nodes.
            include_directives (bool): include `barrier`, `snapshot` etc.

        Returns:
            list[DAGOpNode]: the list of node ids containing the given op.
        """
    nodes = []
    for node in self._multi_graph.nodes():
        if isinstance(node, DAGOpNode):
            if not include_directives and getattr(node.op, '_directive', False):
                continue
            if op is None or isinstance(node.op, op):
                nodes.append(node)
    return nodes