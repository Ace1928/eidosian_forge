import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_weight_bf_path(self):
    G = nx.DiGraph()
    G.add_nodes_from('abcd')
    G.add_edge('a', 'd', weight=0)
    G.add_edge('a', 'b', weight=1)
    G.add_edge('b', 'c', weight=-3)
    G.add_edge('c', 'd', weight=1)
    assert nx.bellman_ford_path(G, 'a', 'd') == ['a', 'b', 'c', 'd']
    assert nx.bellman_ford_path_length(G, 'a', 'd') == -1