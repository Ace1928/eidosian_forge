import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance_upper_bound(self):
    G1 = circular_ladder_graph(2)
    G2 = circular_ladder_graph(6)
    assert graph_edit_distance(G1, G2, upper_bound=5) is None
    assert graph_edit_distance(G1, G2, upper_bound=24) == 22
    assert graph_edit_distance(G1, G2) == 22