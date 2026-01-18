import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_multigraph_keys_max(self):
    """Tests that the maximum spanning edges of a multigraph
        preserves edge keys.
        """
    G = nx.MultiGraph()
    G.add_edge(0, 1, key='a', weight=2)
    G.add_edge(0, 1, key='b', weight=1)
    max_edges = nx.maximum_spanning_edges
    mst_edges = max_edges(G, algorithm=self.algo, data=False)
    assert edges_equal([(0, 1, 'a')], list(mst_edges))