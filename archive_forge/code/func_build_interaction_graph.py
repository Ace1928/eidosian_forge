from collections import defaultdict
import statistics
import random
import numpy as np
from rustworkx import PyDiGraph, PyGraph, connected_components
from qiskit.circuit import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag
from qiskit._accelerate import vf2_layout
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.error_map import ErrorMap
def build_interaction_graph(dag, strict_direction=True):
    """Build an interaction graph from a dag."""
    im_graph = PyDiGraph(multigraph=False) if strict_direction else PyGraph(multigraph=False)
    im_graph_node_map = {}
    reverse_im_graph_node_map = {}

    class MultiQEncountered(Exception):
        """Used to singal an error-status return from the DAG visitor."""

    def _visit(dag, weight, wire_map):
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                if isinstance(node.op, ForLoopOp):
                    inner_weight = len(node.op.params[0]) * weight
                else:
                    inner_weight = weight
                for block in node.op.blocks:
                    inner_wire_map = {inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)}
                    _visit(circuit_to_dag(block), inner_weight, inner_wire_map)
                continue
            len_args = len(node.qargs)
            qargs = [wire_map[q] for q in node.qargs]
            if len_args == 1:
                if qargs[0] not in im_graph_node_map:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph_node_map[qargs[0]] = im_graph.add_node(weights)
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                else:
                    im_graph[im_graph_node_map[qargs[0]]][node.op.name] += weight
            if len_args == 2:
                if qargs[0] not in im_graph_node_map:
                    im_graph_node_map[qargs[0]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                if qargs[1] not in im_graph_node_map:
                    im_graph_node_map[qargs[1]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[1]]] = qargs[1]
                edge = (im_graph_node_map[qargs[0]], im_graph_node_map[qargs[1]])
                if im_graph.has_edge(*edge):
                    im_graph.get_edge_data(*edge)[node.name] += weight
                else:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph.add_edge(*edge, weights)
            if len_args > 2:
                raise MultiQEncountered()
    try:
        _visit(dag, 1, {bit: bit for bit in dag.qubits})
    except MultiQEncountered:
        return None
    free_nodes = {}
    if not strict_direction:
        conn_comp = connected_components(im_graph)
        for comp in conn_comp:
            if len(comp) == 1:
                index = comp.pop()
                free_nodes[index] = im_graph[index]
                im_graph.remove_node(index)
    return (im_graph, im_graph_node_map, reverse_im_graph_node_map, free_nodes)