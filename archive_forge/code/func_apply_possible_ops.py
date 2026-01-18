import itertools
from typing import (
import numpy as np
import networkx as nx
from cirq import circuits, ops, value
import cirq.contrib.acquaintance as cca
from cirq.contrib import circuitdag
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph
def apply_possible_ops(self) -> int:
    """Applies all logical operations possible given the current mapping."""
    nodes = list(self.remaining_dag.findall_nodes_until_blocked(self.acts_on_nonadjacent_qubits))
    assert not any((self.remaining_dag.has_edge(b, a) for a, b in itertools.combinations(nodes, 2)))
    assert not any((self.acts_on_nonadjacent_qubits(node.val) for node in nodes))
    remaining_nodes = [node for node in self.remaining_dag.ordered_nodes() if node not in nodes]
    for node, remaining_node in itertools.product(nodes, remaining_nodes):
        assert not self.remaining_dag.has_edge(remaining_node, node)
    for node in nodes:
        self.remaining_dag.remove_node(node)
        logical_op = node.val
        physical_op = logical_op.with_qubits(*self.log_to_phys(*logical_op.qubits))
        assert len(physical_op.qubits) < 2 or physical_op.qubits in self.device_graph.edges
        self.physical_ops.append(physical_op)
    return len(nodes)