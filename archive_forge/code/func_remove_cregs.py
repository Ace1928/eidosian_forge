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
def remove_cregs(self, *cregs):
    """
        Remove classical registers from the circuit, leaving underlying bits
        in place.

        Raises:
            DAGCircuitError: a creg is not a ClassicalRegister, or is not in
            the circuit.
        """
    if any((not isinstance(creg, ClassicalRegister) for creg in cregs)):
        raise DAGCircuitError('cregs not of type ClassicalRegister: %s' % [r for r in cregs if not isinstance(r, ClassicalRegister)])
    unknown_cregs = set(cregs).difference(self.cregs.values())
    if unknown_cregs:
        raise DAGCircuitError('cregs not in circuit: %s' % unknown_cregs)
    for creg in cregs:
        del self.cregs[creg.name]
        for j in range(creg.size):
            bit = creg[j]
            bit_position = self._clbit_indices[bit]
            bit_position.registers.remove((creg, j))