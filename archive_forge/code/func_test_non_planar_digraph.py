import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_non_planar_digraph(self):
    G = nx.DiGraph(nx.complete_graph(5))
    G.remove_edge(1, 2)
    G.remove_edge(4, 1)
    self.check_graph(G, is_planar=False)