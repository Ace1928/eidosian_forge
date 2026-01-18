import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_identity_digraph_array(self):
    """Conversion from digraph to array to digraph."""
    A = nx.to_numpy_array(self.G2)
    self.identity_conversion(self.G2, A, nx.DiGraph())