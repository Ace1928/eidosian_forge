import pytest
import networkx as nx
from networkx import barbell_graph
from networkx.algorithms.community import modularity, partition_quality
from networkx.algorithms.community.quality import inter_community_edges
def test_good_partition(self):
    """Tests that a good partition has a high coverage measure."""
    G = barbell_graph(3, 0)
    partition = [{0, 1, 2}, {3, 4, 5}]
    assert 6 / 7 == pytest.approx(partition_quality(G, partition)[0], abs=1e-07)