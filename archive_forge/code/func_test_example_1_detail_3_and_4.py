import pytest
import networkx as nx
from networkx.algorithms.approximation import k_components
from networkx.algorithms.approximation.kcomponents import _AntiGraph, _same
def test_example_1_detail_3_and_4():
    G = graph_example_1()
    result = k_components(G)
    assert len(result[3]) == 8
    assert len([c for c in result[3] if len(c) == 15]) == 4
    assert len([c for c in result[3] if len(c) == 5]) == 4
    assert len(result[4]) == 8
    assert all((len(c) == 5 for c in result[4]))
    for k, components in result.items():
        if k < 3:
            continue
        for component in components:
            K = nx.node_connectivity(G.subgraph(component))
            assert K >= k