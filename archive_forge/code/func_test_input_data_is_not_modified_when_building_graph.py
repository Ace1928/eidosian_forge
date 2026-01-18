import copy
import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import cytoscape_data, cytoscape_graph
def test_input_data_is_not_modified_when_building_graph():
    G = nx.path_graph(4)
    input_data = cytoscape_data(G)
    orig_data = copy.deepcopy(input_data)
    cytoscape_graph(input_data)
    assert input_data == orig_data