import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_st_node_cut
from networkx.utils import arbitrary_element
def test_node_cutset_exception():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (3, 4)])
    for flow_func in flow_funcs:
        pytest.raises(nx.NetworkXError, nx.minimum_node_cut, G, flow_func=flow_func)