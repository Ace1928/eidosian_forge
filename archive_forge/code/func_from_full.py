import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
@classmethod
def from_full(cls, num_qubits, bidirectional=True) -> 'CouplingMap':
    """Return a fully connected coupling map on n qubits."""
    cmap = cls(description='full')
    if bidirectional:
        cmap.graph = rx.generators.directed_mesh_graph(num_qubits)
    else:
        edge_list = []
        for i in range(num_qubits):
            for j in range(i):
                edge_list.append((j, i))
        cmap.graph.extend_from_edge_list(edge_list)
    return cmap