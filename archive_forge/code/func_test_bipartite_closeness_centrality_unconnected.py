import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_closeness_centrality_unconnected(self):
    G = nx.complete_bipartite_graph(3, 3)
    G.add_edge(6, 7)
    c = bipartite.closeness_centrality(G, [0, 2, 4, 6], normalized=False)
    answer = {0: 10.0 / 7, 2: 10.0 / 7, 4: 10.0 / 7, 6: 10.0, 1: 10.0 / 7, 3: 10.0 / 7, 5: 10.0 / 7, 7: 10.0}
    assert c == answer