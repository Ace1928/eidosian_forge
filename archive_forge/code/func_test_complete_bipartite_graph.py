import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
def test_complete_bipartite_graph(self):
    G = nx.complete_bipartite_graph(6, 9)
    assert g6.to_graph6_bytes(G, header=False) == b'N??F~z{~Fw^_~?~?^_?\n'