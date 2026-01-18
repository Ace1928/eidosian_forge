import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
def make_symmetric(self):
    """
        Convert uni-directional edges into bi-directional.
        """
    edges = self.get_edges()
    edge_set = set(edges)
    for src, dest in edges:
        if (dest, src) not in edge_set:
            self.graph.add_edge(dest, src, None)
    self._dist_matrix = None
    self._is_symmetric = None