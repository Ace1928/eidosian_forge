import io
import time
import pytest
import networkx as nx
def test_key_raises(self):
    s = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version=\'1.2\'>\n    <graph mode="static" defaultedgetype="directed" name="">\n        <nodes>\n            <node id="0" label="Hello">\n              <attvalues>\n                <attvalue for=\'0\' value=\'1\'/>\n              </attvalues>\n            </node>\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1" type="undirected"/>\n        </edges>\n    </graph>\n</gexf>\n'
    fh = io.BytesIO(s.encode('UTF-8'))
    pytest.raises(nx.NetworkXError, nx.read_gexf, fh)