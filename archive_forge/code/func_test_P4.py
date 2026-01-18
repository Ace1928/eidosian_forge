import math
from functools import partial
import pytest
import networkx as nx
def test_P4(self):
    G = nx.path_graph(4)
    self.test(G, [(0, 2)], [(0, 2, 0.5)])