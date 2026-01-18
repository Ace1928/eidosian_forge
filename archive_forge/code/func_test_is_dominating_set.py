import pytest
import networkx as nx
def test_is_dominating_set():
    G = nx.path_graph(4)
    d = {1, 3}
    assert nx.is_dominating_set(G, d)
    d = {0, 2}
    assert nx.is_dominating_set(G, d)
    d = {1}
    assert not nx.is_dominating_set(G, d)