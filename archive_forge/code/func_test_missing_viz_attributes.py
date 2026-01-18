import io
import time
import pytest
import networkx as nx
def test_missing_viz_attributes(self):
    G = nx.Graph()
    G.add_node(0, label='1', color='green')
    G.nodes[0]['viz'] = {'size': 54}
    G.nodes[0]['viz']['position'] = {'x': 0, 'y': 1, 'z': 0}
    G.nodes[0]['viz']['color'] = {'r': 0, 'g': 0, 'b': 256}
    G.nodes[0]['viz']['shape'] = 'http://random.url'
    G.nodes[0]['viz']['thickness'] = 2
    fh = io.BytesIO()
    nx.write_gexf(G, fh, version='1.1draft')
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))
    fh = io.BytesIO()
    nx.write_gexf(G, fh, version='1.2draft')
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert H.nodes[0]['viz']['color']['a'] == 1.0
    G = nx.Graph()
    G.add_node(0, label='1', color='green')
    G.nodes[0]['viz'] = {'size': 54}
    G.nodes[0]['viz']['position'] = {'x': 0, 'y': 1, 'z': 0}
    G.nodes[0]['viz']['color'] = {'r': 0, 'g': 0, 'b': 256, 'a': 0.5}
    G.nodes[0]['viz']['shape'] = 'ftp://random.url'
    G.nodes[0]['viz']['thickness'] = 2
    fh = io.BytesIO()
    nx.write_gexf(G, fh)
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))