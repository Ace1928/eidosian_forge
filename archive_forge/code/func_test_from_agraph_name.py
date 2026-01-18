import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_agraph_name(self):
    G = nx.Graph(name='test')
    A = nx.nx_agraph.to_agraph(G)
    H = nx.nx_agraph.from_agraph(A)
    assert G.name == 'test'