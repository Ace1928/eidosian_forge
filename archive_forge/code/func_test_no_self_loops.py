import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_no_self_loops(self):
    """Tests for forbidding self-loops."""
    n = 10
    k = 3
    G = random_uniform_k_out_graph(n, k, self_loops=False)
    assert nx.number_of_selfloops(G) == 0
    assert all((d == k for v, d in G.out_degree()))