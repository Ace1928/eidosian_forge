import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
@classmethod
def from_grid(cls, num_rows, num_columns, bidirectional=True) -> 'CouplingMap':
    """Return a coupling map of qubits connected on a grid of num_rows x num_columns."""
    cmap = cls(description='grid')
    cmap.graph = rx.generators.directed_grid_graph(num_rows, num_columns, bidirectional=bidirectional)
    return cmap