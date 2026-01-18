import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_nodes_missing_node_from_position():
    G = nx.path_graph(3)
    pos = {0: (0, 0), 1: (1, 1)}
    with pytest.raises(nx.NetworkXError, match='has no position'):
        nx.draw_networkx_nodes(G, pos)