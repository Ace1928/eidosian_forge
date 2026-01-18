import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_find_cliques2(self):
    hcl = list(nx.find_cliques(self.H))
    assert sorted(map(sorted, hcl)) == [[1, 2], [1, 4, 5, 6], [2, 3], [3, 4, 6]]