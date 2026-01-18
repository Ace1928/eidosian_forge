from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
def compute_periodicity_cycle_basis(self) -> None:
    """Compute periodicity vectors of the connected component."""
    simple_graph = nx.Graph(self._connected_subgraph)
    cycles = nx.cycle_basis(simple_graph)
    all_deltas: list[list] = []
    for cyc in map(list, cycles):
        cyc.append(cyc[0])
        this_cycle_deltas = [np.zeros(3, int)]
        for node1, node2 in [(node1, cyc[inode1 + 1]) for inode1, node1 in enumerate(cyc[:-1])]:
            this_cycle_deltas_new = []
            for edge_data in self._connected_subgraph[node1][node2].values():
                delta = get_delta(node1, node2, edge_data)
                for current_delta in this_cycle_deltas:
                    this_cycle_deltas_new.append(current_delta + delta)
            this_cycle_deltas = this_cycle_deltas_new
        all_deltas.extend(this_cycle_deltas)
        all_deltas = get_linearly_independent_vectors(all_deltas)
        if len(all_deltas) == 3:
            return
    edges = simple_graph.edges()
    for n1, n2 in edges:
        if n1 == n2:
            continue
        if len(self._connected_subgraph[n1][n2]) == 1:
            continue
        if len(self._connected_subgraph[n1][n2]) > 1:
            for iedge1, iedge2 in itertools.combinations(self._connected_subgraph[n1][n2], 2):
                e1data = self._connected_subgraph[n1][n2][iedge1]
                e2data = self._connected_subgraph[n1][n2][iedge2]
                current_delta = get_delta(n1, n2, e1data)
                delta = get_delta(n2, n1, e2data)
                current_delta += delta
                all_deltas.append(current_delta)
        else:
            raise ValueError('Should not be here ...')
        all_deltas = get_linearly_independent_vectors(all_deltas)
        if len(all_deltas) == 3:
            self._periodicity_vectors = all_deltas
            return
    self._periodicity_vectors = all_deltas