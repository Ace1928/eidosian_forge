import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_is_empty():
    graphs = [nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()]
    for G in graphs:
        assert nx.is_empty(G)
        G.add_nodes_from(range(5))
        assert nx.is_empty(G)
        G.add_edges_from([(1, 2), (3, 4)])
        assert not nx.is_empty(G)