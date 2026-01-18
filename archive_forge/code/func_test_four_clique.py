import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_four_clique():
    paths = [(11, 12, 13, 14, 11, 13, 14, 12), (21, 22, 23, 24, 21, 23, 24, 22), (100, 13), (12, 100, 22), (13, 200, 23), (14, 300, 24)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    local_ccs = fset(nx.k_edge_components(G, k=3))
    subgraphs = fset(nx.k_edge_subgraphs(G, k=3))
    assert local_ccs != subgraphs
    clique1 = frozenset(paths[0])
    clique2 = frozenset(paths[1])
    assert clique1.union(clique2).union({100}) in local_ccs
    assert clique1 in subgraphs
    assert clique2 in subgraphs
    assert G.degree(100) == 3
    _check_edge_connectivity(G)