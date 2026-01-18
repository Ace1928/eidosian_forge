from itertools import combinations
import pytest
import networkx as nx
def test_invalid_nodes_raise_error(asia_graph):
    """
    Test that graphs that have invalid nodes passed in raise errors.
    """
    with pytest.raises(nx.NodeNotFound):
        nx.d_separated(asia_graph, {0}, {1}, {2})
    with pytest.raises(nx.NodeNotFound):
        nx.is_minimal_d_separator(asia_graph, 0, 1, {2})
    with pytest.raises(nx.NodeNotFound):
        nx.minimal_d_separator(asia_graph, 0, 1)