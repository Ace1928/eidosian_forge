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
def substitute_node_with_dag(self, node, input_dag, wires=None, propagate_condition=True):
    """Replace one node with dag.

        Args:
            node (DAGOpNode): node to substitute
            input_dag (DAGCircuit): circuit that will substitute the node
            wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
                in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
                and the order gets matched to the node wires by qargs first, then cargs, then
                conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
                that the ``node`` acts on.
            propagate_condition (bool): If ``True`` (default), then any ``condition`` attribute on
                the operation within ``node`` is propagated to each node in the ``input_dag``.  If
                ``False``, then the ``input_dag`` is assumed to faithfully implement suitable
                conditional logic already.  This is ignored for :class:`.ControlFlowOp`\\ s (i.e.
                treated as if it is ``False``); replacements of those must already fulfill the same
                conditional logic or this function would be close to useless for them.

        Returns:
            dict: maps node IDs from `input_dag` to their new node incarnations in `self`.

        Raises:
            DAGCircuitError: if met with unexpected predecessor/successors
        """
    if not isinstance(node, DAGOpNode):
        raise DAGCircuitError(f'expected node DAGOpNode, got {type(node)}')
    if isinstance(wires, dict):
        wire_map = wires
    else:
        wires = input_dag.wires if wires is None else wires
        node_cargs = set(node.cargs)
        node_wire_order = list(node.qargs) + list(node.cargs)
        if not propagate_condition and self._operation_may_have_bits(node.op):
            node_wire_order += [bit for bit in self._bits_in_operation(node.op) if bit not in node_cargs]
        if len(wires) != len(node_wire_order):
            raise DAGCircuitError(f'bit mapping invalid: expected {len(node_wire_order)}, got {len(wires)}')
        wire_map = dict(zip(wires, node_wire_order))
        if len(wire_map) != len(node_wire_order):
            raise DAGCircuitError('bit mapping invalid: some bits have duplicate entries')
    for input_dag_wire, our_wire in wire_map.items():
        if our_wire not in self.input_map:
            raise DAGCircuitError(f'bit mapping invalid: {our_wire} is not in this DAG')
        check_type = Qubit if isinstance(our_wire, Qubit) else Clbit
        if not isinstance(input_dag_wire, check_type):
            raise DAGCircuitError(f'bit mapping invalid: {input_dag_wire} and {our_wire} are different bit types')
    reverse_wire_map = {b: a for a, b in wire_map.items()}
    if propagate_condition and (not isinstance(node.op, ControlFlowOp)) and ((op_condition := getattr(node.op, 'condition', None)) is not None):
        in_dag = input_dag.copy_empty_like()
        target, value = op_condition
        if isinstance(target, Clbit):
            new_target = reverse_wire_map.get(target, Clbit())
            if new_target not in wire_map:
                in_dag.add_clbits([new_target])
                wire_map[new_target], reverse_wire_map[target] = (target, new_target)
            target_cargs = {new_target}
        else:
            mapped_bits = [reverse_wire_map.get(bit, Clbit()) for bit in target]
            for ours, theirs in zip(target, mapped_bits):
                wire_map[theirs], reverse_wire_map[ours] = (ours, theirs)
            new_target = ClassicalRegister(bits=mapped_bits)
            in_dag.add_creg(new_target)
            target_cargs = set(new_target)
        new_condition = (new_target, value)
        for in_node in input_dag.topological_op_nodes():
            if getattr(in_node.op, 'condition', None) is not None:
                raise DAGCircuitError('cannot propagate a condition to an element that already has one')
            if target_cargs.intersection(in_node.cargs):
                raise DAGCircuitError('cannot propagate a condition to an element that acts on those bits')
            new_op = copy.copy(in_node.op)
            if new_condition:
                if not isinstance(new_op, ControlFlowOp):
                    new_op = new_op.c_if(*new_condition)
                else:
                    new_op.condition = new_condition
            in_dag.apply_operation_back(new_op, in_node.qargs, in_node.cargs, check=False)
    else:
        in_dag = input_dag
    if in_dag.global_phase:
        self.global_phase += in_dag.global_phase
    for in_dag_wire, self_wire in wire_map.items():
        input_node = in_dag.input_map[in_dag_wire]
        output_node = in_dag.output_map[in_dag_wire]
        if in_dag._multi_graph.has_edge(input_node._node_id, output_node._node_id):
            pred = self._multi_graph.find_predecessors_by_edge(node._node_id, lambda edge, wire=self_wire: edge == wire)[0]
            succ = self._multi_graph.find_successors_by_edge(node._node_id, lambda edge, wire=self_wire: edge == wire)[0]
            self._multi_graph.add_edge(pred._node_id, succ._node_id, self_wire)

    def filter_fn(node):
        if not isinstance(node, DAGOpNode):
            return False
        for qarg in node.qargs:
            if qarg not in wire_map:
                return False
        return True

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

    def edge_weight_map(wire):
        return wire_map[wire]
    node_map = self._multi_graph.substitute_node_with_subgraph(node._node_id, in_dag._multi_graph, edge_map_fn, filter_fn, edge_weight_map)
    self._decrement_op(node.op)
    variable_mapper = _classical_resource_map.VariableMapper(self.cregs.values(), wire_map, self.add_creg)
    for old_node_index, new_node_index in node_map.items():
        old_node = in_dag._multi_graph[old_node_index]
        if isinstance(old_node.op, SwitchCaseOp):
            m_op = SwitchCaseOp(variable_mapper.map_target(old_node.op.target), old_node.op.cases_specifier(), label=old_node.op.label)
        elif getattr(old_node.op, 'condition', None) is not None:
            m_op = old_node.op
            if not isinstance(old_node.op, ControlFlowOp):
                new_condition = variable_mapper.map_condition(m_op.condition)
                if new_condition is not None:
                    m_op = m_op.c_if(*new_condition)
            else:
                m_op.condition = variable_mapper.map_condition(m_op.condition)
        else:
            m_op = old_node.op
        m_qargs = [wire_map[x] for x in old_node.qargs]
        m_cargs = [wire_map[x] for x in old_node.cargs]
        new_node = DAGOpNode(m_op, qargs=m_qargs, cargs=m_cargs, dag=self)
        new_node._node_id = new_node_index
        self._multi_graph[new_node_index] = new_node
        self._increment_op(new_node.op)
    return {k: self._multi_graph[v] for k, v in node_map.items()}