import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_five_clique():
    G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    paths = [(1, 100, 6), (2, 100, 7), (3, 200, 8), (4, 200, 100)]
    G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    assert min(dict(nx.degree(G)).values()) == 4
    assert fset(nx.k_edge_components(G, k=3)) == fset(nx.k_edge_subgraphs(G, k=3))
    assert fset(nx.k_edge_components(G, k=4)) != fset(nx.k_edge_subgraphs(G, k=4))
    assert fset(nx.k_edge_components(G, k=5)) != fset(nx.k_edge_subgraphs(G, k=5))
    assert fset(nx.k_edge_components(G, k=6)) == fset(nx.k_edge_subgraphs(G, k=6))
    _check_edge_connectivity(G)