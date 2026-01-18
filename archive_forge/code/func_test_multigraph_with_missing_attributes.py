import io
import time
import pytest
import networkx as nx
def test_multigraph_with_missing_attributes(self):
    G = nx.MultiGraph()
    G.add_node(0, label='1', color='green')
    G.add_node(1, label='2', color='green')
    G.add_edge(0, 1, id='0', weight=3, type='undirected', start=0, end=1)
    G.add_edge(0, 1, id='1', label='foo', start=0, end=1)
    G.add_edge(0, 1)
    fh = io.BytesIO()
    nx.write_gexf(G, fh)
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))