import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_edge_cycle_custom_weight_key(self):
    d = nx.DiGraph()
    d.add_edge('a', 'b', w=-2)
    d.add_edge('b', 'a', w=-1)
    assert nx.negative_edge_cycle(d, weight='w')