import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_core_number2(self):
    core = nx.core_number(self.H)
    nodes_by_core = [sorted((n for n in core if core[n] == val)) for val in range(3)]
    assert nodes_equal(nodes_by_core[0], [0])
    assert nodes_equal(nodes_by_core[1], [1, 3])
    assert nodes_equal(nodes_by_core[2], [2, 4, 5, 6])