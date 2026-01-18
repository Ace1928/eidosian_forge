import pytest
import networkx as nx
def test_center_parameter(self):
    G = nx.path_graph(1)
    nx.random_layout(G, center=(1, 1))
    vpos = nx.circular_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)
    vpos = nx.planar_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)
    vpos = nx.spring_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)
    vpos = nx.fruchterman_reingold_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)
    vpos = nx.spectral_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)
    vpos = nx.shell_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)
    vpos = nx.spiral_layout(G, center=(1, 1))
    assert tuple(vpos[0]) == (1, 1)