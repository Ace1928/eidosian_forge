import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_edge_subgraph(self):
    assert self.G.edge_subgraph([(1, 2), (0, 3)]).adj == nx.edge_subgraph(self.G, [(1, 2), (0, 3)]).adj
    assert self.DG.edge_subgraph([(1, 2), (0, 3)]).adj == nx.edge_subgraph(self.DG, [(1, 2), (0, 3)]).adj