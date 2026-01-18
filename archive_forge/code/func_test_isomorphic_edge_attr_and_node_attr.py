import pytest
import networkx as nx
from networkx.generators import directed
def test_isomorphic_edge_attr_and_node_attr():
    """
    Isomorphic graphs with differing node attributes should yield different graph
    hashes if the 'node_attr' and 'edge_attr' argument is supplied and populated in
    the graph, and there are no hash collisions.
    The output should still be invariant to node-relabeling
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G1 = nx.erdos_renyi_graph(n, p * i, seed=500 + i)
        for u in G1.nodes():
            G1.nodes[u]['node_attr1'] = f'{u}-1'
            G1.nodes[u]['node_attr2'] = f'{u}-2'
        for a, b in G1.edges:
            G1[a][b]['edge_attr1'] = f'{a}-{b}-1'
            G1[a][b]['edge_attr2'] = f'{a}-{b}-2'
        g1_hash_edge1_node1 = nx.weisfeiler_lehman_graph_hash(G1, edge_attr='edge_attr1', node_attr='node_attr1')
        g1_hash_edge2_node2 = nx.weisfeiler_lehman_graph_hash(G1, edge_attr='edge_attr2', node_attr='node_attr2')
        g1_hash_edge1_node2 = nx.weisfeiler_lehman_graph_hash(G1, edge_attr='edge_attr1', node_attr='node_attr2')
        g1_hash_no_attr = nx.weisfeiler_lehman_graph_hash(G1)
        assert g1_hash_edge1_node1 != g1_hash_no_attr
        assert g1_hash_edge2_node2 != g1_hash_no_attr
        assert g1_hash_edge1_node1 != g1_hash_edge2_node2
        assert g1_hash_edge1_node2 != g1_hash_edge2_node2
        assert g1_hash_edge1_node2 != g1_hash_edge1_node1
        G2 = nx.relabel_nodes(G1, {u: -1 * u for u in G1.nodes()})
        g2_hash_edge1_node1 = nx.weisfeiler_lehman_graph_hash(G2, edge_attr='edge_attr1', node_attr='node_attr1')
        g2_hash_edge2_node2 = nx.weisfeiler_lehman_graph_hash(G2, edge_attr='edge_attr2', node_attr='node_attr2')
        assert g1_hash_edge1_node1 == g2_hash_edge1_node1
        assert g1_hash_edge2_node2 == g2_hash_edge2_node2