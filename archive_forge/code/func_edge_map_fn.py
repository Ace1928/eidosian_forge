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
def edge_map_fn(source, _target, self_wire):
    wire = reverse_wire_map[self_wire]
    if source == node._node_id:
        wire_output_id = in_dag.output_map[wire]._node_id
        out_index = in_dag._multi_graph.predecessor_indices(wire_output_id)[0]
        if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
            return None
    else:
        wire_input_id = in_dag.input_map[wire]._node_id
        out_index = in_dag._multi_graph.successor_indices(wire_input_id)[0]
        if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
            return None
    return out_index