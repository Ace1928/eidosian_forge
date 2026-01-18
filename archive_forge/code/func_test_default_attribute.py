import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_default_attribute(self):
    G = nx.Graph(name='Fred')
    G.add_node(1, label=1, color='green')
    nx.add_path(G, [0, 1, 2, 3])
    G.add_edge(1, 2, weight=3)
    G.graph['node_default'] = {'color': 'yellow'}
    G.graph['edge_default'] = {'weight': 7}
    fh = io.BytesIO()
    self.writer(G, fh)
    fh.seek(0)
    H = nx.read_graphml(fh, node_type=int)
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    assert G.graph == H.graph