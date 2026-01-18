import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_directed_aux_graph():
    a, b, c, d, e, f, g, h, i = 'abcdefghi'
    dipaths = [(a, d, b, f, c), (a, e, b), (a, e, b, c, g, b, a), (c, b), (f, g, f), (h, i)]
    G = nx.DiGraph(it.chain(*[pairwise(path) for path in dipaths]))
    aux_graph = EdgeComponentAuxGraph.construct(G)
    components_1 = fset(aux_graph.k_edge_subgraphs(k=1))
    target_1 = fset([{a, b, c, d, e, f, g}, {h}, {i}])
    assert target_1 == components_1
    alt_1 = fset(nx.strongly_connected_components(G))
    assert alt_1 == components_1
    components_2 = fset(aux_graph.k_edge_subgraphs(k=2))
    target_2 = fset([{i}, {e}, {d}, {b, c, f, g}, {h}, {a}])
    assert target_2 == components_2
    components_3 = fset(aux_graph.k_edge_subgraphs(k=3))
    target_3 = fset([{a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}, {i}])
    assert target_3 == components_3