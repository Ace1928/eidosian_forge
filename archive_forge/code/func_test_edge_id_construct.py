import io
import time
import pytest
import networkx as nx
def test_edge_id_construct(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1, {'id': 0}), (1, 2, {'id': 2}), (2, 3)])
    expected = f'<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n  <meta lastmodifieddate="{time.strftime('%Y-%m-%d')}">\n    <creator>NetworkX {nx.__version__}</creator>\n  </meta>\n  <graph defaultedgetype="undirected" mode="static" name="">\n    <nodes>\n      <node id="0" label="0" />\n      <node id="1" label="1" />\n      <node id="2" label="2" />\n      <node id="3" label="3" />\n    </nodes>\n    <edges>\n      <edge source="0" target="1" id="0" />\n      <edge source="1" target="2" id="2" />\n      <edge source="2" target="3" id="1" />\n    </edges>\n  </graph>\n</gexf>'
    obtained = '\n'.join(nx.generate_gexf(G))
    assert expected == obtained