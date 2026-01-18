import copy
import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import adjacency_data, adjacency_graph
from networkx.utils import graphs_equal
def test_adjacency_form_json_serialisable(self):
    G = nx.path_graph(4)
    H = adjacency_graph(json.loads(json.dumps(adjacency_data(G))))
    assert graphs_equal(G, H)