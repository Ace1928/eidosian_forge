import pytest
import networkx as nx
from networkx.algorithms.approximation import diameter
def test_directed_path_graph(self):
    """Test a directed path graph with 10 nodes."""
    graph = nx.path_graph(10).to_directed()
    assert diameter(graph) == 9