from itertools import combinations
import pytest
import networkx as nx
def test_collider_graph_dsep(collider_graph):
    """Example-based test of d-separation for collider_graph."""
    assert nx.d_separated(collider_graph, {0}, {1}, {})
    assert not nx.d_separated(collider_graph, {0}, {1}, {2})