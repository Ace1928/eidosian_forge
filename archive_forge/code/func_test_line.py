import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_line(self):
    G = nx.path_graph(5)
    solution = nx.path_graph(6)
    H = nx.inverse_line_graph(G)
    assert nx.is_isomorphic(H, solution)