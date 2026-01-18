import pytest
import networkx as nx
def test_hierarchy_exception():
    G = nx.cycle_graph(5)
    pytest.raises(nx.NetworkXError, nx.flow_hierarchy, G)