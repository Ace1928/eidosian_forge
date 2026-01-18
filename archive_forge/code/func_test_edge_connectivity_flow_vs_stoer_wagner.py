import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_edge_connectivity_flow_vs_stoer_wagner():
    graph_funcs = [nx.icosahedral_graph, nx.octahedral_graph, nx.dodecahedral_graph]
    for graph_func in graph_funcs:
        G = graph_func()
        assert nx.stoer_wagner(G)[0] == nx.edge_connectivity(G)