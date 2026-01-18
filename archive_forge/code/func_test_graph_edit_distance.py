import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance(self):
    G0 = nx.Graph()
    G1 = path_graph(6)
    G2 = cycle_graph(6)
    G3 = wheel_graph(7)
    assert graph_edit_distance(G0, G0) == 0
    assert graph_edit_distance(G0, G1) == 11
    assert graph_edit_distance(G1, G0) == 11
    assert graph_edit_distance(G0, G2) == 12
    assert graph_edit_distance(G2, G0) == 12
    assert graph_edit_distance(G0, G3) == 19
    assert graph_edit_distance(G3, G0) == 19
    assert graph_edit_distance(G1, G1) == 0
    assert graph_edit_distance(G1, G2) == 1
    assert graph_edit_distance(G2, G1) == 1
    assert graph_edit_distance(G1, G3) == 8
    assert graph_edit_distance(G3, G1) == 8
    assert graph_edit_distance(G2, G2) == 0
    assert graph_edit_distance(G2, G3) == 7
    assert graph_edit_distance(G3, G2) == 7
    assert graph_edit_distance(G3, G3) == 0