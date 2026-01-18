import pytest
import networkx as nx
from networkx.generators import directed
def test_digest_size():
    """
    The hash string lengths should be as expected for a variety of graphs and
    digest sizes
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G = nx.erdos_renyi_graph(n, p * i, seed=1000 + i)
        h16 = nx.weisfeiler_lehman_graph_hash(G)
        h32 = nx.weisfeiler_lehman_graph_hash(G, digest_size=32)
        assert h16 != h32
        assert len(h16) == 16 * 2
        assert len(h32) == 32 * 2