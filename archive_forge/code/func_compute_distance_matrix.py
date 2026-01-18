import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
def compute_distance_matrix(self):
    """Compute the full distance matrix on pairs of nodes.

        The distance map self._dist_matrix is computed from the graph using
        all_pairs_shortest_path_length. This is normally handled internally
        by the :attr:`~qiskit.transpiler.CouplingMap.distance_matrix`
        attribute or the :meth:`~qiskit.transpiler.CouplingMap.distance` method
        but can be called if you're accessing the distance matrix outside of
        those or want to pre-generate it.
        """
    if self._dist_matrix is None:
        self._dist_matrix = rx.digraph_distance_matrix(self.graph, as_undirected=True, null_value=math.inf)