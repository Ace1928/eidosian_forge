import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_icosahedral_disjoint_paths():
    G = nx.icosahedral_graph()
    for flow_func in flow_funcs:
        kwargs = {'flow_func': flow_func}
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        edge_dpaths = list(nx.edge_disjoint_paths(G, 0, 6, **kwargs))
        assert are_edge_disjoint_paths(G, edge_dpaths), errmsg
        assert 5 == len(edge_dpaths), errmsg
        node_dpaths = list(nx.node_disjoint_paths(G, 0, 6, **kwargs))
        assert are_node_disjoint_paths(G, node_dpaths), errmsg
        assert 5 == len(node_dpaths), errmsg