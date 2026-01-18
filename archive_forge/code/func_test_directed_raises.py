import pytest
import networkx as nx
def test_directed_raises(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        dir_G = nx.gn_graph(n=5)
        prev_cc = None
        edge = self.pick_add_edge(dir_G)
        insert = True
        nx.incremental_closeness_centrality(dir_G, edge, prev_cc, insert)