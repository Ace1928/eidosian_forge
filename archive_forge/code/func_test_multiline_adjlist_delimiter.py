import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_multiline_adjlist_delimiter(self):
    fh = io.BytesIO()
    G = nx.path_graph(3)
    nx.write_multiline_adjlist(G, fh, delimiter=':')
    fh.seek(0)
    H = nx.read_multiline_adjlist(fh, nodetype=int, delimiter=':')
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))