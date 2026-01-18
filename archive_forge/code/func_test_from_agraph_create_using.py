import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize('graph_class', (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
def test_from_agraph_create_using(self, graph_class):
    G = nx.path_graph(3)
    A = nx.nx_agraph.to_agraph(G)
    H = nx.nx_agraph.from_agraph(A, create_using=graph_class)
    assert isinstance(H, graph_class)