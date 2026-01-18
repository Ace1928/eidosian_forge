from itertools import combinations
import pytest
import networkx as nx
def test_fork_graph_dsep(fork_graph):
    """Example-based test of d-separation for fork_graph."""
    assert nx.d_separated(fork_graph, {1}, {2}, {0})
    assert not nx.d_separated(fork_graph, {1}, {2}, {})