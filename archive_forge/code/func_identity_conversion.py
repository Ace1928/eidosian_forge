import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def identity_conversion(self, G, A, create_using):
    assert A.sum() > 0
    GG = nx.from_numpy_array(A, create_using=create_using)
    self.assert_equal(G, GG)
    GW = nx.to_networkx_graph(A, create_using=create_using)
    self.assert_equal(G, GW)
    GI = nx.empty_graph(0, create_using).__class__(A)
    self.assert_equal(G, GI)