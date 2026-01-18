import pytest
import networkx as nx
from networkx.generators import directed
def test_isomorphic_subgraph_hash():
    """
    the subgraph hashes should be invariant to node-relabeling when the output is reindexed
    by the same mapping and all hashes don't collide.
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G1 = nx.erdos_renyi_graph(n, p * i, seed=200 + i)
        G2 = nx.relabel_nodes(G1, {u: -1 * u for u in G1.nodes()})
        g1_subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(G1)
        g2_subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(G2)
        assert g1_subgraph_hashes == {-1 * k: v for k, v in g2_subgraph_hashes.items()}