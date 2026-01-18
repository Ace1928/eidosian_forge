import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_empty_digraph(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        bytesIO = io.BytesIO()
        bipartite.write_edgelist(nx.DiGraph(), bytesIO)