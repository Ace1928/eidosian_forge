import pytest
import networkx as nx
def test_max_level():
    G = nx.LFR_benchmark_graph(250, 3, 1.5, 0.009, average_degree=5, min_community=20, seed=10)
    parts_iter = nx.community.louvain_partitions(G, seed=42)
    for max_level, expected in enumerate(parts_iter, 1):
        partition = nx.community.louvain_communities(G, max_level=max_level, seed=42)
        assert partition == expected
    assert max_level > 1
    partition = nx.community.louvain_communities(G, max_level=max_level + 1, seed=42)
    assert partition == expected
    with pytest.raises(ValueError, match='max_level argument must be a positive integer'):
        nx.community.louvain_communities(G, max_level=0)