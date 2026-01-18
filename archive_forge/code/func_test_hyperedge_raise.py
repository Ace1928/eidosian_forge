import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_hyperedge_raise(self):
    s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <hyperedge id="e0" source="n0" target="n2">\n       <endpoint node="n0"/>\n       <endpoint node="n1"/>\n       <endpoint node="n2"/>\n    </hyperedge>\n  </graph>\n</graphml>\n'
    fh = io.BytesIO(s.encode('UTF-8'))
    pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
    pytest.raises(nx.NetworkXError, nx.parse_graphml, s)