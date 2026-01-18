import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_core_number(self):
    core = nx.core_number(self.G)
    nodes_by_core = [sorted((n for n in core if core[n] == val)) for val in range(4)]
    assert nodes_equal(nodes_by_core[0], [21])
    assert nodes_equal(nodes_by_core[1], [17, 18, 19, 20])
    assert nodes_equal(nodes_by_core[2], [9, 10, 11, 12, 13, 14, 15, 16])
    assert nodes_equal(nodes_by_core[3], [1, 2, 3, 4, 5, 6, 7, 8])