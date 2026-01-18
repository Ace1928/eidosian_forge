import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_non_numeric_ordering():
    G = MultiDiGraph([('a', 'b'), ('b', 'c'), ('c', 'a')])
    s = scale_free_graph(3, initial_graph=G)
    assert len(s) == 3
    assert len(s.edges) == 3