import pytest
import networkx as nx
def test_cubical_graph_is_2():
    G = nx.cubical_graph()
    assert nx.bipartite.maximal_extendability(G) == 2