import pytest
import networkx as nx
def test_center_wrong_dimensions(self):
    G = nx.path_graph(1)
    assert id(nx.spring_layout) == id(nx.fruchterman_reingold_layout)
    pytest.raises(ValueError, nx.random_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.circular_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.planar_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.spring_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.spring_layout, G, dim=3, center=(1, 1))
    pytest.raises(ValueError, nx.spectral_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.spectral_layout, G, dim=3, center=(1, 1))
    pytest.raises(ValueError, nx.shell_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.spiral_layout, G, center=(1, 1, 1))
    pytest.raises(ValueError, nx.kamada_kawai_layout, G, center=(1, 1, 1))