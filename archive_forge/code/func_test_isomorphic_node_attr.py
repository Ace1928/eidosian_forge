import pytest
import networkx as nx
from networkx.generators import directed
def test_isomorphic_node_attr():
    """
    Isomorphic graphs with differing node attributes should yield different graph
    hashes if the 'node_attr' argument is supplied and populated in the graph, and
    there are no hash collisions.
    The output should still be invariant to node-relabeling
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G1 = nx.erdos_renyi_graph(n, p * i, seed=400 + i)
        for u in G1.nodes():
            G1.nodes[u]['node_attr1'] = f'{u}-1'
            G1.nodes[u]['node_attr2'] = f'{u}-2'
        g1_hash_with_node_attr1 = nx.weisfeiler_lehman_graph_hash(G1, node_attr='node_attr1')
        g1_hash_with_node_attr2 = nx.weisfeiler_lehman_graph_hash(G1, node_attr='node_attr2')
        g1_hash_no_node_attr = nx.weisfeiler_lehman_graph_hash(G1, node_attr=None)
        assert g1_hash_with_node_attr1 != g1_hash_no_node_attr
        assert g1_hash_with_node_attr2 != g1_hash_no_node_attr
        assert g1_hash_with_node_attr1 != g1_hash_with_node_attr2
        G2 = nx.relabel_nodes(G1, {u: -1 * u for u in G1.nodes()})
        g2_hash_with_node_attr1 = nx.weisfeiler_lehman_graph_hash(G2, node_attr='node_attr1')
        g2_hash_with_node_attr2 = nx.weisfeiler_lehman_graph_hash(G2, node_attr='node_attr2')
        assert g1_hash_with_node_attr1 == g2_hash_with_node_attr1
        assert g1_hash_with_node_attr2 == g2_hash_with_node_attr2