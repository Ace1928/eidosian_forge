import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_multigraph_keys(self):
    s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault="directed">\n    <node id="n0"/>\n    <node id="n1"/>\n    <edge id="e0" source="n0" target="n1"/>\n    <edge id="e1" source="n0" target="n1"/>\n  </graph>\n</graphml>\n'
    fh = io.BytesIO(s.encode('UTF-8'))
    G = nx.read_graphml(fh)
    expected = [('n0', 'n1', 'e0'), ('n0', 'n1', 'e1')]
    assert sorted(G.edges(keys=True)) == expected
    fh.seek(0)
    H = nx.parse_graphml(s)
    assert sorted(H.edges(keys=True)) == expected