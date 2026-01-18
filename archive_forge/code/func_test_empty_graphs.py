import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_st_node_cut
from networkx.utils import arbitrary_element
def test_empty_graphs():
    G = nx.Graph()
    D = nx.DiGraph()
    for interface_func in [nx.minimum_node_cut, nx.minimum_edge_cut]:
        for flow_func in flow_funcs:
            pytest.raises(nx.NetworkXPointlessConcept, interface_func, G, flow_func=flow_func)
            pytest.raises(nx.NetworkXPointlessConcept, interface_func, D, flow_func=flow_func)