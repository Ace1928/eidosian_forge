import pytest
import networkx as nx
from networkx.utils import pairwise
def test_bidirectional_dijkstra_multigraph(self):
    G = nx.MultiGraph()
    G.add_edge('a', 'b', weight=10)
    G.add_edge('a', 'b', weight=100)
    dp = nx.bidirectional_dijkstra(G, 'a', 'b')
    assert dp == (10, ['a', 'b'])