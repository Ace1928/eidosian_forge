import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_multigraph_to_graph(self):
    G = nx.MultiGraph()
    G.add_edges_from([('a', 'b', 2), ('b', 'c', 3)])
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname)
    H = nx.read_graphml(fname)
    assert not H.is_multigraph()
    H = nx.read_graphml(fname, force_multigraph=True)
    assert H.is_multigraph()
    os.close(fd)
    os.unlink(fname)
    G.add_edge('a', 'b', 'e-id')
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname)
    H = nx.read_graphml(fname)
    assert H.is_multigraph()
    H = nx.read_graphml(fname, force_multigraph=True)
    assert H.is_multigraph()
    os.close(fd)
    os.unlink(fname)