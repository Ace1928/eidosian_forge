import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_custom2(self):
    """Case of equal nodes."""
    G = nx.complete_graph(4)
    self.test(G, 0, 0, [1, 2, 3])