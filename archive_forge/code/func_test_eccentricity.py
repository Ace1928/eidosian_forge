from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_eccentricity(self):
    assert nx.eccentricity(self.G, 1) == 6
    e = nx.eccentricity(self.G)
    assert e[1] == 6
    sp = dict(nx.shortest_path_length(self.G))
    e = nx.eccentricity(self.G, sp=sp)
    assert e[1] == 6
    e = nx.eccentricity(self.G, v=1)
    assert e == 6
    e = nx.eccentricity(self.G, v=[1, 1])
    assert e[1] == 6
    e = nx.eccentricity(self.G, v=[1, 2])
    assert e[1] == 6
    G = nx.path_graph(1)
    e = nx.eccentricity(G)
    assert e[0] == 0
    e = nx.eccentricity(G, v=0)
    assert e == 0
    pytest.raises(nx.NetworkXError, nx.eccentricity, G, 1)
    G = nx.empty_graph()
    e = nx.eccentricity(G)
    assert e == {}