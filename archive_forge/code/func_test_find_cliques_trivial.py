import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_find_cliques_trivial(self):
    G = nx.Graph()
    assert sorted(nx.find_cliques(G)) == []
    assert sorted(nx.find_cliques_recursive(G)) == []