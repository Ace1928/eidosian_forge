import pytest
from networkx import NetworkXError, cycle_graph
from networkx.algorithms.bipartite import complete_bipartite_graph, node_redundancy
def test_no_redundant_nodes():
    G = complete_bipartite_graph(2, 2)
    rc = node_redundancy(G)
    assert all((redundancy == 1 for redundancy in rc.values()))
    rc = node_redundancy(G, (2, 3))
    assert rc == {2: 1.0, 3: 1.0}