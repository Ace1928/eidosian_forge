import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_identity_graph_array(self):
    """Conversion from graph to array to graph."""
    A = nx.to_numpy_array(self.G1)
    self.identity_conversion(self.G1, A, nx.Graph())