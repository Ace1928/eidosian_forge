import io
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_empty_attribute():
    """Tests that a GraphML string with an empty attribute can be parsed
    correctly."""
    s = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n    <graphml>\n      <key id="d1" for="node" attr.name="foo" attr.type="string"/>\n      <key id="d2" for="node" attr.name="bar" attr.type="string"/>\n      <graph>\n        <node id="0">\n          <data key="d1">aaa</data>\n          <data key="d2">bbb</data>\n        </node>\n        <node id="1">\n          <data key="d1">ccc</data>\n          <data key="d2"></data>\n        </node>\n      </graph>\n    </graphml>'
    fh = io.BytesIO(s.encode('UTF-8'))
    G = nx.read_graphml(fh)
    assert G.nodes['0'] == {'foo': 'aaa', 'bar': 'bbb'}
    assert G.nodes['1'] == {'foo': 'ccc', 'bar': ''}