from itertools import combinations
import pytest
import networkx as nx
def test_minimal_d_separator_checks_dsep():
    """Test that is_minimal_d_separator checks for d-separation as well."""
    g = nx.DiGraph()
    g.add_edges_from([('A', 'B'), ('A', 'E'), ('B', 'C'), ('B', 'D'), ('D', 'C'), ('D', 'F'), ('E', 'D'), ('E', 'F')])
    assert not nx.d_separated(g, {'C'}, {'F'}, {'D'})
    assert not nx.is_minimal_d_separator(g, 'C', 'F', {'D'})
    assert not nx.is_minimal_d_separator(g, 'C', 'F', {})