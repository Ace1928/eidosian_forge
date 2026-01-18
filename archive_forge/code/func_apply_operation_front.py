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
def apply_operation_front(self, op, qargs=(), cargs=(), *, check=True):
    """Apply an operation to the input of the circuit.

        Args:
            op (qiskit.circuit.Operation): the operation associated with the DAG node
            qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
            cargs (tuple[Clbit]): cbits that op will be applied to
            check (bool): If ``True`` (default), this function will enforce that the
                :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
                :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
                uphold these invariants itself, but the cost of several checks will be skipped.
                This is most useful when building a new DAG from a source of known-good nodes.
        Returns:
            DAGOpNode: the node for the op that was added to the dag

        Raises:
            DAGCircuitError: if initial nodes connected to multiple out edges
        """
    qargs = tuple(qargs)
    cargs = tuple(cargs)
    if self._operation_may_have_bits(op):
        all_cbits = set(self._bits_in_operation(op)).union(cargs)
    else:
        all_cbits = cargs
    if check:
        self._check_condition(op.name, getattr(op, 'condition', None))
        self._check_bits(qargs, self.input_map)
        self._check_bits(all_cbits, self.input_map)
    node = DAGOpNode(op=op, qargs=qargs, cargs=cargs, dag=self)
    node._node_id = self._multi_graph.add_node(node)
    self._increment_op(op)
    self._multi_graph.insert_node_on_out_edges_multiple(node._node_id, [self.input_map[bit]._node_id for bits in (qargs, all_cbits) for bit in bits])
    return node