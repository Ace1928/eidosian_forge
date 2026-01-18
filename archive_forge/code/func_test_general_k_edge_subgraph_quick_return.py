import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_general_k_edge_subgraph_quick_return():
    G = nx.Graph()
    G.add_node(0)
    subgraphs = list(general_k_edge_subgraphs(G, k=1))
    assert len(subgraphs) == 1
    for subgraph in subgraphs:
        assert subgraph.number_of_nodes() == 1
    G.add_node(1)
    subgraphs = list(general_k_edge_subgraphs(G, k=1))
    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        assert subgraph.number_of_nodes() == 1