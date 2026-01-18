import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_long_attribute_type(self):
    s = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key attr.name="cudfversion" attr.type="long" for="node" id="d6" />\n  <graph edgedefault="directed">\n    <node id="n1">\n      <data key="d6">4284</data>\n    </node>\n  </graph>\n</graphml>'
    fh = io.BytesIO(s.encode('UTF-8'))
    G = nx.read_graphml(fh)
    expected = [('n1', {'cudfversion': 4284})]
    assert sorted(G.nodes(data=True)) == expected
    fh.seek(0)
    H = nx.parse_graphml(s)
    assert sorted(H.nodes(data=True)) == expected