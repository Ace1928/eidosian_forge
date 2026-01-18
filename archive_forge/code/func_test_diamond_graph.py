import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_diamond_graph(self):
    G = nx.diamond_graph()
    for edge in G.edges:
        cell = line._select_starting_cell(G, starting_edge=edge)
        assert len(cell) == 3
        assert all((v in G[u] for u in cell for v in cell if u != v))