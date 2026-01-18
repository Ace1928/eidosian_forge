import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_S4(self):
    G = nx.star_graph(4)
    self.test(G, 1, 2, [0])