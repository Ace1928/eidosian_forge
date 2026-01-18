import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
@classmethod
def from_ring(cls, num_qubits, bidirectional=True) -> 'CouplingMap':
    """Return a coupling map of n qubits connected to each of their neighbors in a ring."""
    cmap = cls(description='ring')
    cmap.graph = rx.generators.directed_cycle_graph(num_qubits, bidirectional=bidirectional)
    return cmap