import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
@classmethod
def from_hexagonal_lattice(cls, rows, cols, bidirectional=True) -> 'CouplingMap':
    """Return a hexagonal lattice graph coupling map.

        Args:
            rows (int): The number of rows to generate the graph with.
            cols (int): The number of columns to generate the graph with.
            bidirectional (bool): Whether the edges in the output coupling
                graph are bidirectional or not. By default this is set to
                ``True``
        Returns:
            CouplingMap: A hexagonal lattice coupling graph
        """
    cmap = cls(description='hexagonal-lattice')
    cmap.graph = rx.generators.directed_hexagonal_lattice_graph(rows, cols, bidirectional=bidirectional)
    return cmap