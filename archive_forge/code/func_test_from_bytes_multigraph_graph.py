import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_from_bytes_multigraph_graph(self):
    graph_data = b':An'
    G = nx.from_sparse6_bytes(graph_data)
    assert type(G) == nx.Graph
    multigraph_data = b':Ab'
    M = nx.from_sparse6_bytes(multigraph_data)
    assert type(M) == nx.MultiGraph