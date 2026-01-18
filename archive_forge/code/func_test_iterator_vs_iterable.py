from itertools import chain, combinations
import pytest
import networkx as nx
def test_iterator_vs_iterable():
    G = nx.empty_graph('a')
    assert list(nx.community.label_propagation_communities(G)) == [{'a'}]
    for community in nx.community.label_propagation_communities(G):
        assert community == {'a'}
    pytest.raises(TypeError, next, nx.community.label_propagation_communities(G))