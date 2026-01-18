import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_find_cliques3(self):
    cl = list(nx.find_cliques(self.G, [2]))
    rcl = nx.find_cliques_recursive(self.G, [2])
    expected = [[2, 6, 1, 3], [2, 6, 4]]
    assert sorted(map(sorted, rcl)) == sorted(map(sorted, expected))
    assert sorted(map(sorted, cl)) == sorted(map(sorted, expected))
    cl = list(nx.find_cliques(self.G, [2, 3]))
    rcl = nx.find_cliques_recursive(self.G, [2, 3])
    expected = [[2, 6, 1, 3]]
    assert sorted(map(sorted, rcl)) == sorted(map(sorted, expected))
    assert sorted(map(sorted, cl)) == sorted(map(sorted, expected))
    cl = list(nx.find_cliques(self.G, [2, 6, 4]))
    rcl = nx.find_cliques_recursive(self.G, [2, 6, 4])
    expected = [[2, 6, 4]]
    assert sorted(map(sorted, rcl)) == sorted(map(sorted, expected))
    assert sorted(map(sorted, cl)) == sorted(map(sorted, expected))
    cl = list(nx.find_cliques(self.G, [2, 6, 4]))
    rcl = nx.find_cliques_recursive(self.G, [2, 6, 4])
    expected = [[2, 6, 4]]
    assert sorted(map(sorted, rcl)) == sorted(map(sorted, expected))
    assert sorted(map(sorted, cl)) == sorted(map(sorted, expected))
    with pytest.raises(ValueError):
        list(nx.find_cliques(self.G, [2, 6, 4, 1]))
    with pytest.raises(ValueError):
        list(nx.find_cliques_recursive(self.G, [2, 6, 4, 1]))