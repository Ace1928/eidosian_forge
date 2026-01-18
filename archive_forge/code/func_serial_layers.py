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
def serial_layers(self):
    """Yield a layer for all gates of this circuit.

        A serial layer is a circuit with one gate. The layers have the
        same structure as in layers().
        """
    for next_node in self.topological_op_nodes():
        new_layer = self.copy_empty_like()
        support_list = []
        op = copy.copy(next_node.op)
        qargs = copy.copy(next_node.qargs)
        cargs = copy.copy(next_node.cargs)
        new_layer.apply_operation_back(op, qargs, cargs, check=False)
        if not getattr(next_node.op, '_directive', False):
            support_list.append(list(qargs))
        l_dict = {'graph': new_layer, 'partition': support_list}
        yield l_dict