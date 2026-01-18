import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def agraph_checks(self, G):
    G = self.build_graph(G)
    A = nx.nx_agraph.to_agraph(G)
    H = nx.nx_agraph.from_agraph(A)
    self.assert_equal(G, H)
    fd, fname = tempfile.mkstemp()
    nx.drawing.nx_agraph.write_dot(H, fname)
    Hin = nx.nx_agraph.read_dot(fname)
    self.assert_equal(H, Hin)
    os.close(fd)
    os.unlink(fname)
    fd, fname = tempfile.mkstemp()
    with open(fname, 'w') as fh:
        nx.drawing.nx_agraph.write_dot(H, fh)
    with open(fname) as fh:
        Hin = nx.nx_agraph.read_dot(fh)
    os.close(fd)
    os.unlink(fname)
    self.assert_equal(H, Hin)