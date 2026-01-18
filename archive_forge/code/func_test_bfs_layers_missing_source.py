from functools import partial
import pytest
import networkx as nx
def test_bfs_layers_missing_source(self):
    with pytest.raises(nx.NetworkXError):
        next(nx.bfs_layers(self.G, sources='abc'))
    with pytest.raises(nx.NetworkXError):
        next(nx.bfs_layers(self.G, sources=['abc']))