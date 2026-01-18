import pytest
import networkx as nx
from networkx.generators import directed
def test_isomorphic_edge_attr_subgraph_hash():
    """
    Isomorphic graphs with differing edge attributes should yield different subgraph
    hashes if the 'edge_attr' argument is supplied and populated in the graph, and
    all hashes don't collide.
    The output should still be invariant to node-relabeling
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G1 = nx.erdos_renyi_graph(n, p * i, seed=300 + i)
        for a, b in G1.edges:
            G1[a][b]['edge_attr1'] = f'{a}-{b}-1'
            G1[a][b]['edge_attr2'] = f'{a}-{b}-2'
        g1_hash_with_edge_attr1 = nx.weisfeiler_lehman_subgraph_hashes(G1, edge_attr='edge_attr1')
        g1_hash_with_edge_attr2 = nx.weisfeiler_lehman_subgraph_hashes(G1, edge_attr='edge_attr2')
        g1_hash_no_edge_attr = nx.weisfeiler_lehman_subgraph_hashes(G1, edge_attr=None)
        assert g1_hash_with_edge_attr1 != g1_hash_no_edge_attr
        assert g1_hash_with_edge_attr2 != g1_hash_no_edge_attr
        assert g1_hash_with_edge_attr1 != g1_hash_with_edge_attr2
        G2 = nx.relabel_nodes(G1, {u: -1 * u for u in G1.nodes()})
        g2_hash_with_edge_attr1 = nx.weisfeiler_lehman_subgraph_hashes(G2, edge_attr='edge_attr1')
        g2_hash_with_edge_attr2 = nx.weisfeiler_lehman_subgraph_hashes(G2, edge_attr='edge_attr2')
        assert g1_hash_with_edge_attr1 == {-1 * k: v for k, v in g2_hash_with_edge_attr1.items()}
        assert g1_hash_with_edge_attr2 == {-1 * k: v for k, v in g2_hash_with_edge_attr2.items()}