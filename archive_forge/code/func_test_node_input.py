from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_node_input(self):
    G = nx.grid_graph([range(7, 9), range(3, 6)])
    assert len(G) == 2 * 3
    assert nx.is_isomorphic(G, nx.grid_graph([2, 3]))