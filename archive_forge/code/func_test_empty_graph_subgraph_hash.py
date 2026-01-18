import pytest
import networkx as nx
from networkx.generators import directed
def test_empty_graph_subgraph_hash():
    """ "
    empty graphs should give empty dict subgraph hashes regardless of other params
    """
    G = nx.empty_graph()
    subgraph_hashes1 = nx.weisfeiler_lehman_subgraph_hashes(G)
    subgraph_hashes2 = nx.weisfeiler_lehman_subgraph_hashes(G, edge_attr='edge_attr')
    subgraph_hashes3 = nx.weisfeiler_lehman_subgraph_hashes(G, node_attr='edge_attr')
    subgraph_hashes4 = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=2)
    subgraph_hashes5 = nx.weisfeiler_lehman_subgraph_hashes(G, digest_size=64)
    assert subgraph_hashes1 == {}
    assert subgraph_hashes2 == {}
    assert subgraph_hashes3 == {}
    assert subgraph_hashes4 == {}
    assert subgraph_hashes5 == {}