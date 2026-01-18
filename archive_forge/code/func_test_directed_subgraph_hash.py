import pytest
import networkx as nx
from networkx.generators import directed
def test_directed_subgraph_hash():
    """
    A directed graph with no bi-directional edges should yield different subgraph hashes
    to the same graph taken as undirected, if all hashes don't collide.
    """
    r = 10
    for i in range(r):
        G_directed = nx.gn_graph(10 + r, seed=100 + i)
        G_undirected = nx.to_undirected(G_directed)
        directed_subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(G_directed)
        undirected_subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(G_undirected)
        assert directed_subgraph_hashes != undirected_subgraph_hashes