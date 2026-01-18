import pytest
import networkx as nx
from networkx.utils import pairwise
def test_find_negative_cycle_no_cycle(self):
    G = nx.path_graph(5, create_using=nx.DiGraph())
    pytest.raises(nx.NetworkXError, nx.find_negative_cycle, G, 3)