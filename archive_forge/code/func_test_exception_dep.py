import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
def test_exception_dep(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.MultiDiGraph()
        node_link_data(G, name='node', source='node', target='node', key='node')