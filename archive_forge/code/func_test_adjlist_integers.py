import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_adjlist_integers(self):
    fd, fname = tempfile.mkstemp()
    G = nx.convert_node_labels_to_integers(self.G)
    nx.write_adjlist(G, fname)
    H = nx.read_adjlist(fname, nodetype=int)
    H2 = nx.read_adjlist(fname, nodetype=int)
    assert H is not H2
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))
    os.close(fd)
    os.unlink(fname)