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
def shuffle_coupling_graph(coupling_map, seed, strict_direction=True):
    """Create a shuffled coupling graph from a coupling map."""
    if strict_direction:
        cm_graph = coupling_map.graph
    else:
        cm_graph = coupling_map.graph.to_undirected(multigraph=False)
    cm_nodes = list(cm_graph.node_indexes())
    if seed != -1:
        random.Random(seed).shuffle(cm_nodes)
        shuffled_cm_graph = type(cm_graph)()
        shuffled_cm_graph.add_nodes_from(cm_nodes)
        new_edges = [(cm_nodes[edge[0]], cm_nodes[edge[1]]) for edge in cm_graph.edge_list()]
        shuffled_cm_graph.add_edges_from_no_data(new_edges)
        cm_nodes = [k for k, v in sorted(enumerate(cm_nodes), key=lambda item: item[1])]
        cm_graph = shuffled_cm_graph
    return (cm_graph, cm_nodes)