import pytest
import networkx as nx
from networkx.algorithms.approximation import diameter
def test_complete_undirected_graph(self):
    """Test a complete undirected graph."""
    graph = nx.complete_graph(10)
    assert diameter(graph) == 1