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
def substitute_node(self, node, op, inplace=False, propagate_condition=True):
    """Replace an DAGOpNode with a single operation. qargs, cargs and
        conditions for the new operation will be inferred from the node to be
        replaced. The new operation will be checked to match the shape of the
        replaced operation.

        Args:
            node (DAGOpNode): Node to be replaced
            op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`
                instance to be added to the DAG
            inplace (bool): Optional, default False. If True, existing DAG node
                will be modified to include op. Otherwise, a new DAG node will
                be used.
            propagate_condition (bool): Optional, default True.  If True, a condition on the
                ``node`` to be replaced will be applied to the new ``op``.  This is the legacy
                behaviour.  If either node is a control-flow operation, this will be ignored.  If
                the ``op`` already has a condition, :exc:`.DAGCircuitError` is raised.

        Returns:
            DAGOpNode: the new node containing the added operation.

        Raises:
            DAGCircuitError: If replacement operation was incompatible with
            location of target node.
        """
    if not isinstance(node, DAGOpNode):
        raise DAGCircuitError('Only DAGOpNodes can be replaced.')
    if node.op.num_qubits != op.num_qubits or node.op.num_clbits != op.num_clbits:
        raise DAGCircuitError('Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits).'.format(node.op.num_qubits, node.op.num_clbits, op.num_qubits, op.num_clbits))
    current_wires = {wire for _, _, wire in self.edges(node)}
    new_wires = set(node.qargs) | set(node.cargs)
    if (new_condition := getattr(op, 'condition', None)) is not None:
        new_wires.update(condition_resources(new_condition).clbits)
    elif isinstance(op, SwitchCaseOp):
        if isinstance(op.target, Clbit):
            new_wires.add(op.target)
        elif isinstance(op.target, ClassicalRegister):
            new_wires.update(op.target)
        else:
            new_wires.update(node_resources(op.target).clbits)
    if propagate_condition and (not (isinstance(node.op, ControlFlowOp) or isinstance(op, ControlFlowOp))):
        if new_condition is not None:
            raise DAGCircuitError('Cannot propagate a condition to an operation that already has one.')
        if (old_condition := getattr(node.op, 'condition', None)) is not None:
            if not isinstance(op, Instruction):
                raise DAGCircuitError('Cannot add a condition on a generic Operation.')
            if not isinstance(node.op, ControlFlowOp):
                op = op.c_if(*old_condition)
            else:
                op.condition = old_condition
            new_wires.update(condition_resources(old_condition).clbits)
    if new_wires != current_wires:
        raise DAGCircuitError(f"New operation '{op}' does not span the same wires as the old node '{node}'. New wires: {new_wires}, old wires: {current_wires}.")
    if inplace:
        if op.name != node.op.name:
            self._increment_op(op)
            self._decrement_op(node.op)
        node.op = op
        return node
    new_node = copy.copy(node)
    new_node.op = op
    self._multi_graph[node._node_id] = new_node
    if op.name != node.op.name:
        self._increment_op(op)
        self._decrement_op(node.op)
    return new_node